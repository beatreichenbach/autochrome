import logging
import os.path
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.data.cmfs import CMFS
from autochrome.data.illuminants import ILLUMINANTS_CIE
from autochrome.utils import ocio
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)


LAMBDA_MIN = 390
LAMBDA_MAX = 830
LAMBDA_COUNT = 21

COEFFICIENTS_COUNT = 3


def smoothstep(x: float) -> float:
    return x**2 * (3.0 - 2.0 * x)


def normalize_image(image: Image) -> tuple:
    min_val = np.min(image.array)
    max_val = np.max(image.array)
    image._array = (image.array - min_val) / (max_val - min_val)
    return min_val, max_val


def un_normalize_image(image: Image, min_val: np.ndarray, max_val: np.ndarray) -> None:
    image._array = image.array * (max_val - min_val) + min_val


class SpectralTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
        self.kernels = {}
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = ''
        self.source += f'#define LAMBDA_COUNT {LAMBDA_COUNT}\n'
        self.source += f'#define COEFFICIENTS_COUNT {COEFFICIENTS_COUNT}\n'

        # self.register_dtype('Ray', ray_dtype)

        # self.source += f'__constant int BIN_SIZE = {self.bin_size};\n'
        # self.source += f'__constant int LAMBDA_MIN = {LAMBDA_MIN};\n'
        # self.source += f'__constant int LAMBDA_MAX = {LAMBDA_MAX};\n'
        self.source += self.read_source_file('jakob.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'xyz_to_xyz')

    @lru_cache(1)
    def load_file(self, file: File, resolution: QtCore.QSize) -> Image:
        # load array
        filename = str(file)
        try:
            array = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        except ValueError as e:
            logger.debug(e)
            message = f'Invalid Image path: {filename}'
            raise EngineError(message) from None

        # convert to float32
        if array.dtype == np.uint8:
            array = np.divide(array, 255)
        array = np.float32(array)

        # resize array
        array = cv2.resize(array, (resolution.width(), resolution.height()))

        array = np.dstack((array, np.zeros(array.shape[:2], np.float32)))

        # return image
        image = Image(self.context, array=array, args=(file, resolution))
        return image

    @lru_cache(1)
    def update_model(self) -> Buffer:
        model_path = '/home/beat/dev/autochrome/autochrome/api/tasks/model.npy'
        model_file = File(model_path)
        values = np.load(model_path)
        model = np.zeros(len(values), cl.cltypes.float)
        for i in range(len(values)):
            model[i] = values[i]

        buffer = Buffer(self.context, array=model, args=model_file)
        return buffer

    @lru_cache(1)
    def update_lambdas(
        self, lambda_min: int, lambda_max: int, lambda_count: int
    ) -> Buffer:
        # lambdas = np.linspace(lambda_min, lambda_max, lambda_count)

        lambdas = np.zeros(LAMBDA_COUNT, cl.cltypes.float)
        values = np.linspace(lambda_min, lambda_max, lambda_count)
        for i in range(LAMBDA_COUNT):
            lambdas[i] = values[i]

        args = (lambda_min, lambda_max, lambda_count)
        buffer = Buffer(self.context, array=lambdas, args=args)
        return buffer

    @lru_cache(1)
    def update_cmfs(self, lambdas: Buffer) -> Buffer:
        cmfs_data = CMFS['CIE 2015 2 Degree Standard Observer']
        cmfs_keys = np.array(list(cmfs_data.keys()))
        cmfs_values = np.array(list(cmfs_data.values()))
        cmfs = np.zeros(LAMBDA_COUNT, cl.cltypes.float4)
        for i in range(3):
            values = np.interp(lambdas.array, cmfs_keys, cmfs_values[:, i])
            for j in range(LAMBDA_COUNT):
                cmfs[j][i] = values[j]
        buffer = Buffer(self.context, array=cmfs, args=lambdas)
        return buffer

    @lru_cache(1)
    def update_illuminant(self, lambdas: Buffer) -> Buffer:
        illuminant_data = ILLUMINANTS_CIE['D65']
        illuminant_keys = np.array(list(illuminant_data.keys()))
        illuminant_values = np.array(list(illuminant_data.values()))

        # illuminant = np.interp(lambdas.array, illuminant_keys, illuminant_values)

        illuminant = np.zeros(LAMBDA_COUNT, cl.cltypes.float)
        values = np.interp(lambdas.array, illuminant_keys, illuminant_values)
        for i in range(LAMBDA_COUNT):
            illuminant[i] = values[i]

        buffer = Buffer(self.context, array=illuminant, args=lambdas)
        return buffer

    @lru_cache(1)
    def update_scale(self, resolution: int) -> Buffer:
        values = [
            smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)
        ]
        scale = np.zeros(resolution, cl.cltypes.float)
        for i in range(resolution):
            scale[i] = values[i]

        buffer = Buffer(self.context, array=scale, args=resolution)
        return buffer

    @timer
    @lru_cache(1)
    def spectral_image(
        self,
        image_file: File,
        resolution: QtCore.QSize,
    ) -> Image:
        if self.rebuild:
            self.build()
        image = self.load_file(image_file, resolution)
        # image._array[:, :, :] = np.array([0.18069152, 0.16347412, 0.03151255, 0])
        processor = ocio.colorspace_processor(
            src_name='sRGB - Display', dst_name='CIE-XYZ-D65'
        )
        processor.applyRGBA(image.array)

        min_val, max_val = normalize_image(image)

        model_resolution = 16
        scale = self.update_scale(model_resolution)
        lambdas = self.update_lambdas(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT)
        cmfs = self.update_cmfs(lambdas)
        illuminant = self.update_illuminant(lambdas)
        model = self.update_model()

        # create output buffer
        spectral = self.update_image(resolution)  # , flags=cl.mem_flags.READ_WRITE
        spectral.args = (resolution, model_resolution, image_file)
        # image.clear_image()

        # run program
        self.kernel.set_arg(0, image.image)
        self.kernel.set_arg(1, spectral.image)
        self.kernel.set_arg(2, lambdas.buffer)
        self.kernel.set_arg(3, cmfs.buffer)
        self.kernel.set_arg(4, illuminant.buffer)
        self.kernel.set_arg(5, model.buffer)
        self.kernel.set_arg(6, scale.buffer)
        self.kernel.set_arg(7, np.int32(model_resolution))

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None
        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )
        cl.enqueue_copy(
            self.queue, spectral.array, spectral.image, origin=(0, 0), region=(w, h)
        )

        un_normalize_image(spectral, min_val, max_val)

        logger.debug(spectral.array[16, 16])

        processor = ocio.colorspace_processor(src_name='CIE-XYZ-D65')
        processor.applyRGBA(spectral.array)

        logger.debug(spectral.array[16, 16])

        return spectral

    def run(self, project: Project) -> Image:
        image_file = File(project.input.image_path)
        resolution = project.render.resolution
        image = self.spectral_image(image_file, resolution)
        return image

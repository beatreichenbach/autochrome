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

from autochrome.resources.curves.kodak_ektachrome_100 import SENSITIVITY, DYE_DENSITY

# from autochrome.resources.curves.kodak_portra_800 import SENSITIVITY, DYE_DENSITY
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
    min_val = np.min(image.array[:, :, :3])
    max_val = np.max(image.array[:, :, :3])
    lift = 0.01
    lift = 0
    image._array = (image.array - min_val + lift) / (max_val + lift - min_val)
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

        self.kernels = {
            'xyz_to_sd': cl.Kernel(self.program, 'xyz_to_sd'),
            'xyz_to_xyz': cl.Kernel(self.program, 'xyz_to_xyz'),
            'xyz_to_mask': cl.Kernel(self.program, 'xyz_to_mask'),
        }

    # @lru_cache(1)
    def load_file(self, file: File, resolution: QtCore.QSize) -> Image:
        # load array
        filename = str(file)
        try:
            array = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        except ValueError as e:
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
        model_path = '/home/beat/dev/autochrome/autochrome/api/tasks/model_d65.npy'
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

    def update_spectral_distribution(
        self, resolution: QtCore.QSize, lambda_count: int
    ) -> Buffer:
        shape = (resolution.height(), resolution.width(), lambda_count)
        array = np.zeros(shape, np.float32)
        buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, array.nbytes)
        spectral_distribution = Buffer(self.context, array=array, buffer=buffer)
        return spectral_distribution

    @lru_cache(1)
    def update_spectral_sensitivity(self, lambdas: Buffer) -> Buffer:
        sensitivity_data = SENSITIVITY
        sensitivity_keys = np.array(list(sensitivity_data.keys()))
        sensitivity_values = np.array(list(sensitivity_data.values()))
        sensitivity = np.zeros(LAMBDA_COUNT, cl.cltypes.float4)
        for i in range(3):
            values = np.interp(
                lambdas.array, sensitivity_keys, sensitivity_values[:, i]
            )
            for j in range(LAMBDA_COUNT):
                sensitivity[j][2 - i] = values[j]
        buffer = Buffer(self.context, array=sensitivity, args=lambdas)
        return buffer

    @lru_cache(1)
    def update_spectral_density(self, lambdas: Buffer) -> Buffer:
        density_data = DYE_DENSITY
        density_keys = np.array(list(density_data.keys()))
        density_values = np.array(list(density_data.values()))

        density = np.zeros(LAMBDA_COUNT, cl.cltypes.float)
        values = np.interp(lambdas.array, density_keys, density_values)
        for i in range(LAMBDA_COUNT):
            density[i] = values[i]

        buffer = Buffer(self.context, array=density, args=lambdas)
        return buffer

    @lru_cache(1)
    def update_test_pattern(self, resolution: int) -> Buffer:
        space = np.linspace(0, 1, resolution)
        pattern_3d = np.stack(np.meshgrid(space, space, space), axis=3)

        # size = int(np.ceil(np.sqrt(resolution**3)))
        shape = (resolution, resolution**2, 3)
        pattern_2d = pattern_3d.flatten()
        pattern_2d.resize(shape, refcheck=False)

        pattern = Buffer(context=self.context, array=pattern_2d, args=resolution)
        return pattern

    @timer
    def spectral_image(
        self,
        image_file: File,
        resolution: QtCore.QSize,
    ) -> tuple[Image, ...]:
        if self.rebuild:
            self.build()
        image = self.load_file(image_file, resolution)

        # processor = ocio.colorspace_processor(
        #     src_name='sRGB - Display', dst_name='CIE-XYZ-D65'
        # )
        processor = ocio.colorspace_processor(dst_name='CIE-XYZ-D65')
        processor.applyRGBA(image.array)

        # min_val, max_val = normalize_image(image)
        # logger.debug(f'image_values: {image.array[0, 0]}')

        model_resolution = 16
        scale = self.update_scale(model_resolution)
        lambdas = self.update_lambdas(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT)
        cmfs = self.update_cmfs(lambdas)
        illuminant = self.update_illuminant(lambdas)
        model = self.update_model()

        spectral_sensitivity = self.update_spectral_sensitivity(lambdas)
        spectral_density = self.update_spectral_density(lambdas)

        # create output buffer
        spectral_r = self.update_image(resolution)  # , flags=cl.mem_flags.READ_WRITE
        spectral_r.args = (resolution, model_resolution, image_file)
        spectral_g = self.update_image(resolution)  # , flags=cl.mem_flags.READ_WRITE
        spectral_g.args = (resolution, model_resolution, image_file)
        spectral_b = self.update_image(resolution)  # , flags=cl.mem_flags.READ_WRITE
        spectral_b.args = (resolution, model_resolution, image_file)
        # image.clear_image()

        image.clear_image()
        # run program
        kernel = self.kernels['xyz_to_mask']
        kernel.set_arg(0, image.image)
        kernel.set_arg(1, spectral_r.image)
        kernel.set_arg(2, spectral_g.image)
        kernel.set_arg(3, spectral_b.image)
        kernel.set_arg(4, lambdas.buffer)
        kernel.set_arg(5, cmfs.buffer)
        kernel.set_arg(6, illuminant.buffer)
        kernel.set_arg(7, model.buffer)
        kernel.set_arg(8, scale.buffer)
        kernel.set_arg(9, np.int32(model_resolution))
        kernel.set_arg(10, spectral_sensitivity.buffer)

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None
        cl.enqueue_nd_range_kernel(
            self.queue, kernel, global_work_size, local_work_size
        )
        cl.enqueue_copy(
            self.queue, spectral_r.array, spectral_r.image, origin=(0, 0), region=(w, h)
        )
        cl.enqueue_copy(
            self.queue, spectral_g.array, spectral_g.image, origin=(0, 0), region=(w, h)
        )
        cl.enqueue_copy(
            self.queue, spectral_b.array, spectral_b.image, origin=(0, 0), region=(w, h)
        )

        # un_normalize_image(image, min_val, max_val)
        # processor = ocio.colorspace_processor(src_name='CIE-XYZ-D65')
        # processor.applyRGBA(image.array)

        # image._array[:, :] = (
        #     image._array[:, :] * spectral.array[:, :, 0, np.newaxis]
        #     + image._array[:, :] * spectral.array[:, :, 1, np.newaxis]
        #     + image._array[:, :] * spectral.array[:, :, 2, np.newaxis]
        # )

        # un_normalize_image(spectral, min_val, max_val)
        # spectral.array[:, :, :] = 0.3

        return spectral_r, spectral_g, spectral_b

    def run(self, project: Project) -> tuple[Image, ...]:
        image_file = File(project.input.image_path)
        resolution = project.render.resolution
        image = self.spectral_image(image_file, resolution)
        return image

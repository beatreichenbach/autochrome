import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.jakob import get_scale, get_illuminant, get_cmfs
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.resources.curves.kodak_ektachrome_100 import SENSITIVITY, DYE_DENSITY
from autochrome.resources.curves.kodak_portra_800 import SENSITIVITY, DYE_DENSITY
from autochrome.utils import ocio
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)


LAMBDA_MIN = 390
LAMBDA_MAX = 830
COEFFICIENTS_COUNT = 3

# TODO: remove
LAMBDA_COUNT = 21


def smoothstep(x: float) -> float:
    return x**2 * (3.0 - 2.0 * x)


def normalize_image(image: Image) -> tuple:
    min_val = np.min(image.array[:, :, :3])
    max_val = np.max(image.array[:, :, :3])
    image._array = (image.array - min_val) / (max_val - min_val)
    return min_val, max_val


def un_normalize_image(image: Image, min_val: np.ndarray, max_val: np.ndarray) -> None:
    image._array = image.array * (max_val - min_val) + min_val


class EmulsionTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
        self.kernel = None
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = (
            f'#define LAMBDA_COUNT {LAMBDA_COUNT}\n'
            f'#define COEFFICIENTS_COUNT {COEFFICIENTS_COUNT}\n'
        )
        self.source += self.read_source_file('emulsion.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'emulsion_layers')

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
    def update_model(self, model_file: File) -> Buffer:
        model_path = str(model_file)
        model = np.load(model_path)
        model_cl = np.ravel(model).astype(cl.cltypes.float)
        buffer = Buffer(self.context, array=model_cl, args=model_file)
        return buffer

    @lru_cache(1)
    def update_lambdas(
        self, lambda_min: int, lambda_max: int, lambda_count: int
    ) -> Buffer:
        lambdas = np.linspace(lambda_min, lambda_max, lambda_count)
        lambdas_cl = lambdas.astype(cl.cltypes.float)
        args = (lambda_min, lambda_max, lambda_count)
        buffer = Buffer(self.context, array=lambdas_cl, args=args)
        return buffer

    @lru_cache(1)
    def update_cmfs(self, variation: str, lambdas: Buffer) -> Buffer:
        cmfs = get_cmfs(variation, lambdas.array)
        cmfs_cl = np.zeros(cmfs.shape[0], cl.cltypes.float4)
        for i, xyz in enumerate(cmfs):
            for j in range(3):
                cmfs_cl[i][j] = xyz[j]
        buffer = Buffer(self.context, array=cmfs_cl, args=(variation, lambdas))
        return buffer

    @lru_cache(1)
    def update_illuminant(self, standard_illuminant: str, lambdas: Buffer) -> Buffer:
        illuminant = get_illuminant(standard_illuminant, lambdas.array)
        illuminant_cl = illuminant.astype(cl.cltypes.float)
        args = (standard_illuminant, lambdas)
        buffer = Buffer(self.context, array=illuminant_cl, args=args)
        return buffer

    @lru_cache(1)
    def update_scale(self, resolution: int) -> Buffer:
        scale = np.array(get_scale(resolution), np.float32)
        scale_cl = scale.astype(cl.cltypes.float)
        buffer = Buffer(self.context, array=scale_cl, args=resolution)
        return buffer

    @lru_cache(1)
    def update_spectral_sensitivity(self, curves_file: File, lambdas: Buffer) -> Buffer:
        sensitivity_data = SENSITIVITY
        sensitivity_keys = np.array(list(sensitivity_data.keys()))
        sensitivity_values = np.array(list(sensitivity_data.values()))
        sensitivity = np.zeros(lambdas.shape, cl.cltypes.float4)
        for i in range(3):
            values = np.interp(
                lambdas.array, sensitivity_keys, sensitivity_values[:, i]
            )
            for j in range(LAMBDA_COUNT):
                sensitivity[j][2 - i] = values[j]
        buffer = Buffer(self.context, array=sensitivity, args=(lambdas, curves_file))
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

    @lru_cache(1)
    def update_xyz_image(
        self, image_file: File, input_colorspace: str, resolution: QtCore.QSize
    ) -> Image:
        image = self.load_file(image_file, resolution)

        # NOTE: To ensure values are always between 0 and 1 convert into sRGB space
        processor = ocio.colorspace_processor(
            src_name=input_colorspace, dst_name='Output - sRGB'
        )

        # TODO: handle no processors (WARNING?)
        if processor:
            processor.applyRGBA(image.array)

        processor = ocio.colorspace_processor(dst_name='Utility - XYZ - D60')
        # processor = ocio.colorspace_processor(dst_name='CIE-XYZ-D65')
        if processor:
            processor.applyRGBA(image.array)

        return image

    @timer
    def spectral_images(
        self,
        image_file: File,
        input_colorspace: str,
        resolution: QtCore.QSize,
        model_file: File,
        curves_file: File,
        lambda_count: int,
    ) -> tuple[Image, ...]:
        if self.rebuild:
            self.build()

        standard_illuminant = 'D65'
        cmfs_variation = 'CIE 2015 2 Degree Standard Observer'
        lambda_min = LAMBDA_MIN
        lambda_max = LAMBDA_MAX

        xyz_image = self.update_xyz_image(image_file, input_colorspace, resolution)
        # xyz_image.clear_image()

        # min_val, max_val = normalize_image(image)
        # logger.debug(f'{min_val=}, {max_val=}')

        model = self.update_model(model_file)
        model_resolution = 16

        scale = self.update_scale(model_resolution)
        lambdas = self.update_lambdas(lambda_min, lambda_max, lambda_count)
        cmfs = self.update_cmfs(cmfs_variation, lambdas)
        illuminant = self.update_illuminant(standard_illuminant, lambdas)
        spectral_sensitivity = self.update_spectral_sensitivity(curves_file, lambdas)

        # create output buffer
        spectral_images = []
        for _ in range(3):
            image = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)
            image.args = (xyz_image, model, curves_file)
            spectral_images.append(image)
            # image.clear_image()
        spectral_images = tuple(spectral_images)

        # run program
        self.kernel.set_arg(0, xyz_image.image)
        self.kernel.set_arg(1, spectral_images[0].image)
        self.kernel.set_arg(2, spectral_images[1].image)
        self.kernel.set_arg(3, spectral_images[2].image)
        self.kernel.set_arg(4, model.buffer)
        self.kernel.set_arg(5, np.int32(model_resolution))
        self.kernel.set_arg(6, scale.buffer)
        self.kernel.set_arg(7, lambdas.buffer)
        self.kernel.set_arg(8, cmfs.buffer)
        self.kernel.set_arg(9, illuminant.buffer)
        self.kernel.set_arg(10, spectral_sensitivity.buffer)

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None
        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )
        for image in spectral_images:
            cl.enqueue_copy(
                self.queue, image.array, image.image, origin=(0, 0), region=(w, h)
            )

        return spectral_images

    def run(self, project: Project) -> tuple[Image, ...]:
        image_file = File(project.input.image_path)
        model_file = File(
            '/home/beat/dev/autochrome/autochrome/api/tasks/afef8e3dd5781a7df338b7f65ffcc0ad.npy'
        )
        curves_file = File(
            '/home/beat/dev/autochrome/autochrome/resources/curves/kodak_ektachrome_100.json'
        )
        spectral_images = self.spectral_images(
            image_file=image_file,
            input_colorspace=project.input.colorspace,
            resolution=project.render.resolution,
            model_file=model_file,
            curves_file=curves_file,
            lambda_count=project.emulsion.lambda_count,
        )
        return spectral_images

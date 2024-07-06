import json
import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks import jakob
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.storage import Storage
from autochrome.utils import ocio, color
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)
storage = Storage()


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
        self._lambda_count = 0
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = (
            f'#define LAMBDA_COUNT {self._lambda_count}\n'
            f'#define COEFFICIENTS_COUNT {jakob.COEFFICIENTS_COUNT}\n'
        )
        self.source += self.read_source_file('emulsion.cl')

        super().build(*args, **kwargs)

        self.kernel = cl.Kernel(self.program, 'emulsion_layers')

    @lru_cache(1)
    def load_file(self, file: File, resolution: QtCore.QSize | None = None) -> Image:
        # load array
        filename = str(file)
        try:
            array = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        except ValueError:
            message = f'Invalid Image path: {filename}'
            raise EngineError(message) from None

        # convert to float32
        if array.dtype == np.uint8:
            array = np.divide(array, 255)
        array = np.float32(array)

        # resize array
        if resolution:
            array = cv2.resize(array, (resolution.width(), resolution.height()))

        array = np.dstack((array, np.zeros(array.shape[:2], np.float32)))

        # return image
        image = Image(self.context, array=array, args=(file, resolution))
        return image

    @lru_cache(1)
    def update_model(self, model_path: str) -> Buffer:
        model = np.load(model_path)
        model_cl = np.ravel(model).astype(cl.cltypes.float)
        buffer = Buffer(self.context, array=model_cl, args=model_path)
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
        cmfs = color.get_cmfs(variation, lambdas.array)
        cmfs_cl = np.zeros(cmfs.shape[0], cl.cltypes.float4)
        for i, xyz in enumerate(cmfs):
            for j in range(3):
                cmfs_cl[i][j] = xyz[j]
        buffer = Buffer(self.context, array=cmfs_cl, args=(variation, lambdas))
        return buffer

    @lru_cache(1)
    def update_illuminant(self, standard_illuminant: str, lambdas: Buffer) -> Buffer:
        illuminant = color.get_illuminant(standard_illuminant, lambdas.array)
        illuminant_cl = illuminant.astype(cl.cltypes.float)
        args = (standard_illuminant, lambdas)
        buffer = Buffer(self.context, array=illuminant_cl, args=args)
        return buffer

    @lru_cache(1)
    def update_scale(self, resolution: int) -> Buffer:
        scale = np.array(jakob.get_scale(resolution), np.float32)
        scale_cl = scale.astype(cl.cltypes.float)
        buffer = Buffer(self.context, array=scale_cl, args=resolution)
        return buffer

    @lru_cache(1)
    def update_spectral_sensitivity(self, curves_file: File, lambdas: Buffer) -> Buffer:
        with open(str(curves_file), 'r') as f:
            sensitivity_data = json.load(f)

        lambda_count = lambdas.shape[0]

        sensitivity_keys = np.array(list(map(int, sensitivity_data.keys())))
        sensitivity_values = np.array(list(sensitivity_data.values()))
        sensitivity = np.zeros(lambda_count, cl.cltypes.float4)
        for i in range(3):
            values = np.interp(
                lambdas.array, sensitivity_keys, sensitivity_values[:, i]
            )
            for j in range(lambda_count):
                sensitivity[j][2 - i] = values[j]
        buffer = Buffer(self.context, array=sensitivity, args=(lambdas, curves_file))
        return buffer

    @lru_cache(1)
    def update_xyz_image(
        self,
        image_file: File,
        input_colorspace: str,
        resolution: QtCore.QSize | None = None,
    ) -> Image:
        image = self.load_file(image_file, resolution)

        image_array = image.array.copy()

        # TODO: handle no processors (WARNING?)
        # NOTE: To ensure values are always between 0 and 1 convert into sRGB space
        # dst_name = 'sRGB - Display'
        dst_name = 'Output - sRGB'
        processor = ocio.colorspace_processor(
            src_name=input_colorspace, dst_name=dst_name
        )
        if processor:
            processor.applyRGBA(image_array)

        # dst_name='CIE-XYZ-D65'
        dst_name = 'Utility - XYZ - D60'
        processor = ocio.colorspace_processor(dst_name=dst_name)
        if processor:
            processor.applyRGBA(image_array)

        xyz_image = Image(
            self.context,
            array=image_array,
            args=(image_file, input_colorspace, resolution),
        )

        return xyz_image

    @lru_cache(1)
    def spectral_images(
        self,
        image_file: File,
        input_colorspace: str,
        resolution: QtCore.QSize,
        force_resolution: bool,
        model_path: str,
        curves_file: File,
        lambda_min: int,
        lambda_max: int,
        lambda_count: int,
        model_resolution: int,
        standard_illuminant: str,
        cmfs_variation: str,
    ) -> tuple[Image, ...]:
        lambda_count_changed = lambda_count != self._lambda_count
        if lambda_count_changed:
            self._lambda_count = lambda_count
        if self.rebuild or lambda_count_changed:
            self.build()

        if not force_resolution:
            resolution = None

        xyz_image = self.update_xyz_image(image_file, input_colorspace, resolution)
        if resolution is None:
            resolution = QtCore.QSize(
                xyz_image.array.shape[1], xyz_image.array.shape[0]
            )
        # xyz_image.clear_image()

        # min_val, max_val = normalize_image(image)
        # logger.debug(f'{min_val=}, {max_val=}')

        model = self.update_model(model_path)

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

    @timer
    def run(self, project: Project) -> tuple[Image, ...]:
        image_file = File(project.input.image_path)
        model_path = jakob.get_model_path(project)
        curves_path = storage.decode_path(project.emulsion.curves_file)
        curves_file = File(curves_path)
        spectral_images = self.spectral_images(
            image_file=image_file,
            input_colorspace=project.input.colorspace,
            resolution=project.render.resolution,
            force_resolution=project.render.force_resolution,
            model_path=model_path,
            curves_file=curves_file,
            lambda_min=project.emulsion.lambda_min,
            lambda_max=project.emulsion.lambda_max,
            lambda_count=project.emulsion.wavelength_count,
            model_resolution=project.emulsion.model_resolution,
            standard_illuminant=project.emulsion.standard_illuminant,
            cmfs_variation=project.emulsion.cmfs_variation,
        )
        return spectral_images

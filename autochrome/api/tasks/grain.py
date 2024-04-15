import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.halation import HalationTask
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.api.tasks.spectral import normalize_image, un_normalize_image
from autochrome.utils import ocio

logger = logging.getLogger(__name__)


@lru_cache(1)
def lambda_lut(mu: float, sigma: float, count: int = 256) -> np.ndarray:
    # Generates a LUT for lambda values used by the Poisson distribution.
    # The lut stores the lambda value and the e^-lambda value.
    # Equation (2) in Realistic Film Grain Rendering by Alasdair et al.
    # https://www.ipol.im/pub/art/2017/192/article_lr.pdf

    use_source_code = True
    lut = np.zeros((count), cl.cltypes.float2)

    for i in range(count):
        u = i / count

        if use_source_code:
            cell_size = 1 / np.ceil(1 / mu)
            area = np.pi * ((mu * mu) + (sigma * sigma))
            lambda_u = -((cell_size * cell_size) / area) * np.log(1 - u)
        else:
            mean_radius = np.exp(mu + (sigma**2 / 2))
            mean_area = np.pi * (mean_radius**2)
            lambda_u = (1 / mean_area) * np.log(1 - u)

        lut[i]['x'] = lambda_u
        lut[i]['y'] = np.exp(-lambda_u)

    return lut


class GrainTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
        self.halation_task = HalationTask(self.queue)
        self.kernels = {}
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = ''
        # self.source += f'#define BATCH_PRIMITIVE_COUNT {BATCH_PRIMITIVE_COUNT}\n'

        # self.register_dtype('Ray', ray_dtype)

        # self.source += f'__constant int BIN_SIZE = {self.bin_size};\n'
        # self.source += f'__constant int LAMBDA_MIN = {LAMBDA_MIN};\n'
        # self.source += f'__constant int LAMBDA_MAX = {LAMBDA_MAX};\n'
        self.source += self.read_source_file('rand.cl')
        self.source += self.read_source_file('grain.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'grain')

    @lru_cache(1)
    def load_file(self, file: File, resolution: QtCore.QSize) -> np.ndarray:
        # load array
        filename = str(file)
        try:
            array = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        except ValueError as e:
            logger.debug(e)
            message = f'Invalid Image path for flare light: {filename}'
            raise EngineError(message) from None

        # convert to float32
        if array.dtype == np.uint8:
            array = np.divide(array, 255)
        array = np.float32(array)

        # resize array
        array = cv2.resize(array, (resolution.width(), resolution.height()))

        return array

    @lru_cache(1)
    def update_lambda_lut(self, mu: float, sigma: float, count: int = 256) -> Buffer:
        lut = lambda_lut(mu, sigma, count)
        buffer = Buffer(self.context, array=lut, args=(mu, sigma, count))
        return buffer

    def render_grain(
        self,
        image_file: File,
        resolution: QtCore.QSize,
        grain_mu: float,
        grain_sigma: float,
        blur_sigma: float,
        samples: int,
        seed_offset: int,
        bounds: tuple[float, float, float, float],
        spectrals: tuple[Image, ...],
        spec: Image,
    ) -> Image:
        if self.rebuild:
            self.build()

        bounds = (0, 0, resolution.width(), resolution.height())

        # image_array = self.load_file(image_file, resolution)
        # image_array = np.dstack(
        #     (image_array, np.zeros(image_array.shape[:2], np.float32))
        # )
        # processor = ocio.colorspace_processor(
        #     src_name='sRGB - Display', dst_name='ACEScc'
        # )
        # processor.applyRGBA(image_array)

        # processor = ocio.colorspace_processor(dst_name='ACEScc')
        # processor.applyRGBA(halation.array)

        # min_val, max_val = normalize_image(halation)
        # logger.debug((min_val, max_val))
        # halation.array[:, :, 3] = 0

        # logger.debug(f'image_array_shape: {image_array.shape}')
        # image = Image(self.context, image_array, args=str(image_file))
        # logger.debug(image.shape)
        # min_val, max_val = normalize_image(image)

        lut = self.update_lambda_lut(grain_mu, grain_sigma, count=256)

        # create output buffer
        grain = self.update_image(
            resolution,
            flags=cl.mem_flags.WRITE_ONLY,
            channel_order=cl.channel_order.LUMINANCE,
        )
        grain.args = (
            resolution,
            grain_mu,
            grain_sigma,
            blur_sigma,
            samples,
            seed_offset,
            bounds,
        )

        render_bounds = np.array(bounds, dtype=cl.cltypes.float4)
        # image.clear_image()

        # run program
        # self.kernel.set_arg(0, image.image)
        self.kernel.set_arg(1, grain.image)
        self.kernel.set_arg(2, lut.buffer)
        self.kernel.set_arg(3, np.int32(samples))
        self.kernel.set_arg(4, np.float32(grain_sigma))
        self.kernel.set_arg(5, np.float32(grain_mu))
        self.kernel.set_arg(6, np.float32(blur_sigma))
        self.kernel.set_arg(7, np.int32(seed_offset))
        self.kernel.set_arg(8, render_bounds)

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        # cl.enqueue_nd_range_kernel(
        #     self.queue, self.kernel, global_work_size, local_work_size
        # )
        # cl.enqueue_copy(
        #     self.queue, grain.array, grain.image, origin=(0, 0), region=(w, h)
        # )
        # output = grain.array

        output_array = np.zeros((h, w, 4), np.float32)

        for c, spectral in enumerate(spectrals):
            # output += spectral.array
            # continue
            # if c > 1:
            #     continue
            if c == 0:
                halation = self.halation_task.render(spectral, spec)
            else:
                halation = spectral
            mean = np.mean(halation.array, axis=2)
            image = Image(self.context, array=mean)

            self.kernel.set_arg(0, image.image)

            # 1431655765 is roughly a third of the max range for the random noise.
            self.kernel.set_arg(7, np.int32(seed_offset + c * 1431655765))

            cl.enqueue_nd_range_kernel(
                self.queue, self.kernel, global_work_size, local_work_size
            )
            cl.enqueue_copy(
                self.queue, grain.array, grain.image, origin=(0, 0), region=(w, h)
            )

            output_array += halation.array * (grain.array / mean)[:, :, np.newaxis]
            # output += mean[:, :, np.newaxis]

        processor = ocio.colorspace_processor(src_name='CIE-XYZ-D65')
        processor.applyRGBA(output_array)

        grain = Image(
            self.context,
            output_array,
            args=(
                resolution,
                image_file,
                samples,
                grain_sigma,
                grain_mu,
                blur_sigma,
                seed_offset,
                spectrals,
                spec,
            ),
        )

        # un_normalize_image(grain, min_val, max_val)

        return grain

    def run(self, project: Project, spectrals: tuple[Image, ...], spec: Image) -> Image:
        image_file = File(project.input.image_path)
        bounds = (
            project.grain.bounds_min.x(),
            project.grain.bounds_min.y(),
            project.grain.bounds_max.x(),
            project.grain.bounds_max.y(),
        )
        image = self.render_grain(
            image_file,
            resolution=project.render.resolution,
            grain_mu=project.grain.grain_mu,
            grain_sigma=project.grain.grain_sigma,
            blur_sigma=project.grain.blur_sigma,
            samples=project.grain.samples,
            seed_offset=project.grain.seed_offset,
            bounds=bounds,
            spectrals=spectrals,
            spec=spec,
        )
        return image

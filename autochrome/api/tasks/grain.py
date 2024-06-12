import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.utils import ocio
from autochrome.utils.timing import timer

MAX_INT = 2**31
logger = logging.getLogger(__name__)


@lru_cache(1)
def create_lambda_lut(mu: float, sigma: float, count: int = 256) -> np.ndarray:
    """Generates a LUT for lambda values used by the Poisson distribution.
    The lut stores the lambda value and the e^-lambda value.
    Equation (2) in Realistic Film Grain Rendering by Alasdair et al.
    https://www.ipol.im/pub/art/2017/192/article_lr.pdf
    """

    use_source_code = True
    lut = np.zeros(count, cl.cltypes.float2)

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
        self.kernel = None
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = self.read_source_file('rand.cl')
        self.source += self.read_source_file('grain.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'grain')

    @lru_cache(1)
    def update_lambda_lut(self, mu: float, sigma: float, count: int = 256) -> Buffer:
        lut = create_lambda_lut(mu, sigma, count)
        buffer = Buffer(self.context, array=lut, args=(mu, sigma, count))
        return buffer

    @lru_cache(1)
    def render(
        self,
        spectral_images: tuple[Image, ...],
        samples: int,
        seed_offset: int,
        grain_mu: float,
        grain_sigma: float,
        blur_sigma: float,
        lift: float,
    ) -> Image:
        if self.rebuild:
            self.build()

        h, w = spectral_images[0].shape[:2]
        resolution = QtCore.QSize(w, h)

        render_bounds = np.array((0, 0, w, h), dtype=cl.cltypes.float4)

        lambda_lut = self.update_lambda_lut(grain_mu, grain_sigma, count=256)

        # create output buffer
        grain = self.update_image(
            resolution,
            flags=cl.mem_flags.WRITE_ONLY,
            channel_order=cl.channel_order.LUMINANCE,
        )

        # image.clear_image()

        # run program
        self.kernel.set_arg(1, grain.image)
        self.kernel.set_arg(2, lambda_lut.buffer)
        self.kernel.set_arg(3, np.int32(samples))
        self.kernel.set_arg(4, np.float32(grain_sigma))
        self.kernel.set_arg(5, np.float32(grain_mu))
        self.kernel.set_arg(6, np.float32(blur_sigma))
        self.kernel.set_arg(7, np.int32(seed_offset))
        self.kernel.set_arg(8, render_bounds)

        global_work_size = (w, h)
        local_work_size = None

        grain_array = np.zeros((h, w, 4), np.float32)

        for c, spectral_image in enumerate(spectral_images):
            # grain_array += spectral_image.array
            # continue

            # layer = spectral_image.array.copy()
            # layer += lift
            # layer = np.power(layer, 1 / 2.2)
            # mean = (
            #     layer[:, :, 0] * 0.2126
            #     + layer[:, :, 1] * 0.7152
            #     + layer[:, :, 2] * 0.0722
            # )
            # mean = np.mean(layer, axis=2)
            # mean += 0.02

            luminance = spectral_image.array[:, :, 1] + lift
            EPSILON = 0.00001
            luminance = np.maximum(luminance, EPSILON)

            image = Image(self.context, array=luminance)

            self.kernel.set_arg(0, image.image)
            # NOTE: Set seed to roughly a third of the max range for the random noise.
            seed = np.int32(int((MAX_INT - seed_offset) / 3) * c)
            self.kernel.set_arg(7, seed)

            cl.enqueue_nd_range_kernel(
                self.queue, self.kernel, global_work_size, local_work_size
            )
            cl.enqueue_copy(
                self.queue, grain.array, grain.image, origin=(0, 0), region=(w, h)
            )

            result = spectral_image.array * (grain.array / luminance)[:, :, np.newaxis]
            grain_array += result

        grain = Image(
            self.context,
            grain_array,
            args=(
                spectral_images,
                samples,
                seed_offset,
                grain_mu,
                grain_sigma,
                blur_sigma,
                lift,
            ),
        )

        return grain

    @timer
    def run(self, project: Project, spectral_images: tuple[Image, ...]) -> Image:
        image = self.render(
            spectral_images=spectral_images,
            samples=project.grain.samples,
            seed_offset=project.grain.seed_offset,
            grain_mu=project.grain.grain_mu,
            grain_sigma=project.grain.grain_sigma,
            blur_sigma=project.grain.blur_sigma,
            lift=project.grain.lift,
        )
        return image

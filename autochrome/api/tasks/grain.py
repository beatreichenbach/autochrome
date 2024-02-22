import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer

logger = logging.getLogger(__name__)


class GrainTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
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
    def lambda_lut(self, mu: float, sigma: float, count: int = 256) -> Buffer:
        # Generates a LUT for lambda values used by the Poisson distribution.
        # The lut stores the lambda value and the e^-lambda value.
        # Equation (2) in Realistic Film Grain Rendering by Alasdair et al.
        # https://www.ipol.im/pub/art/2017/192/article_lr.pdf

        use_source_code = True
        lut = np.float32((2, count))
        for i in range(count):
            u = i / count

            if use_source_code:
                ag = 1 / np.ceil(1 / mu)
                area = np.pi * ((mu * mu) + (sigma * sigma))
                lambda_u = -((ag * ag) / area) * np.log(1 - u)
            else:
                mean_radius = np.exp(mu + (sigma**2 / 2))
                mean_area = np.pi * (mean_radius**2)
                lambda_u = (1 / mean_area) * np.log(1 - u)

            lut[0] = lambda_u
            lut[1] = np.exp(-lambda_u)

        return lut

    def render_grain(
        self,
        image_file: File,
        resolution: QtCore.QSize,
        grain_mu: float,
        grain_sigma: float,
        blur_sigma: float,
        samples: int,
    ) -> Image:
        if self.rebuild:
            self.build()

        image_array = self.load_file(image_file, resolution)
        image_array = np.dstack(
            (
                image_array,
                np.zeros((image_array.shape[0], image_array.shape[1]), np.float32),
            )
        )
        image = Image(self.context, image_array, args=str(image_file))
        logger.debug(image.shape)

        # lambda_lut = self.lambda_lut(grain_mu, grain_sigma)

        # create output buffer
        grain = self.update_image(resolution, flags=cl.mem_flags.WRITE_ONLY)
        grain.args = (
            resolution,
            grain_mu,
            grain_sigma,
            blur_sigma,
            samples,
        )

        # image.clear_image()

        # run program
        self.kernel.set_arg(0, image.image)
        self.kernel.set_arg(1, grain.image)
        # self.kernel.set_arg(2, lambda_lut.buffer)
        self.kernel.set_arg(2, np.int32(samples))
        self.kernel.set_arg(3, np.float32(grain_sigma))
        self.kernel.set_arg(4, np.float32(grain_mu))
        self.kernel.set_arg(5, np.float32(blur_sigma))
        # self.kernel.set_arg(6, np.int32(0))
        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None
        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )
        cl.enqueue_copy(
            self.queue, grain.array, grain.image, origin=(0, 0), region=(w, h)
        )

        return grain

    def run(self, project: Project) -> Image:
        image_file = File(project.input.image_path)
        resolution = project.render.resolution
        grain_mu = 0.01
        grain_sigma = 0.1
        blur_sigma = 0.4
        samples = 50
        image = self.render_grain(
            image_file, resolution, grain_mu, grain_sigma, blur_sigma, samples
        )
        return image

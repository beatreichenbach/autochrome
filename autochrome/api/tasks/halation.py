import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.api.tasks.spectral import normalize_image, un_normalize_image
from autochrome.utils import ocio

logger = logging.getLogger(__name__)


class HalationTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
        self.kernel = None
        self.kernel2 = None
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = ''
        self.source += self.read_source_file('halation.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'ConvolveH')
        self.kernel2 = cl.Kernel(self.program, 'ConvolveV')

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

    def render(
        self,
        image_file: File,
        resolution: QtCore.QSize,
        spec: Image,
    ) -> Image:
        if self.rebuild:
            self.build()

        image_array = self.load_file(image_file, resolution)
        image_array = np.dstack(
            (image_array, np.zeros(image_array.shape[:2], np.float32))
        )
        processor = ocio.colorspace_processor(src_name='sRGB - Display')
        processor.applyRGBA(image_array)

        black = 0.25
        white = 0.4
        mask = np.clip((image_array[:, :, 0] - black) / (white - black), 0, 1)
        logger.debug(np.min(image_array[:, :, 0]))
        logger.debug(np.max(image_array[:, :, 0]))

        image = Image(
            self.context, image_array * mask[:, :, np.newaxis], args=str(image_file)
        )

        # create temp buffer
        temp = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)

        # create output buffer
        halation = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)
        halation.args = (resolution, image_file, spec)

        mask_size = int(spec.array.shape[0] / 2)
        mask_array = np.ascontiguousarray(spec.array[mask_size, :, 0])
        mask_array /= np.sum(mask_array)

        mask = Buffer(self.context, array=mask_array, args=(spec,))
        logger.debug(mask_array)

        # run program
        self.kernel.set_arg(0, image.image)
        self.kernel.set_arg(1, temp.image)
        self.kernel.set_arg(2, mask.buffer)
        self.kernel.set_arg(3, np.int32(mask_size))

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )

        # run program
        self.kernel2.set_arg(0, temp.image)
        self.kernel2.set_arg(1, halation.image)
        self.kernel2.set_arg(2, mask.buffer)
        self.kernel2.set_arg(3, np.int32(mask_size))

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel2, global_work_size, local_work_size
        )

        cl.enqueue_copy(
            self.queue, halation.array, halation.image, origin=(0, 0), region=(w, h)
        )

        # image_array[:, :, 0] += halation.array[:, :, 0]

        # max_density = 3
        # S = 10
        # a = 1.1
        # C = 1
        # density = np.log10(1 + S * np.power(image_array, a)) + C
        # image_array = density / max_density

        output = Image(self.context, array=image_array, args=spec)

        return output

    def run(self, project: Project, spec: Image) -> Image:

        image_file = File(project.input.image_path)
        image = self.render(
            image_file=image_file, resolution=project.render.resolution, spec=spec
        )
        return image

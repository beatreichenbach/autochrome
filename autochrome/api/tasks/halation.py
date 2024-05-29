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
        self.source = self.read_source_file('halation.cl')

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
        image: Image,
        spec: Image,
        mask_range: tuple[float, float],
        amount: float,
    ) -> Image:
        if self.rebuild:
            self.build()

        resolution = QtCore.QSize(image.array.shape[1], image.array.shape[0])

        # create temp buffer
        temp1 = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)

        # create output buffer
        temp2 = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)

        mask_size = int(spec.array.shape[0] / 2)
        mask_array = np.ascontiguousarray(spec.array[mask_size, :, 0])
        mask_array /= np.sum(mask_array)

        mask = Buffer(self.context, array=mask_array, args=(spec,))
        # logger.debug(mask_array)

        # input_array = image.array.copy()

        mask_min, mask_max = mask_range
        masked_array = (
            image.array
            * np.clip((image.array[:, :, 1] - mask_min) / (mask_max - mask_min), 0, 1)[
                :, :, np.newaxis
            ]
        )

        processor = ocio.colorspace_processor(src_name='CIE-XYZ-D65')
        processor.applyRGBA(masked_array)

        masked_image = Image(self.context, array=masked_array)

        # run program
        self.kernel.set_arg(0, masked_image.image)
        self.kernel.set_arg(1, temp1.image)
        self.kernel.set_arg(2, mask.buffer)
        self.kernel.set_arg(3, np.int32(mask_size))

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )

        # run program
        self.kernel2.set_arg(0, temp1.image)
        self.kernel2.set_arg(1, temp2.image)
        self.kernel2.set_arg(2, mask.buffer)
        self.kernel2.set_arg(3, np.int32(mask_size))

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel2, global_work_size, local_work_size
        )

        cl.enqueue_copy(
            self.queue, temp2.array, temp2.image, origin=(0, 0), region=(w, h)
        )

        processor = ocio.colorspace_processor(dst_name='CIE-XYZ-D65')
        processor.applyRGBA(temp2.array)

        output_array = np.clip(image.array + temp2.array * amount, 0, 1)

        output = Image(
            self.context, array=output_array, args=(image, spec, mask_range, amount)
        )

        return output

    def run(self, project: Project, image: Image, spec: Image) -> Image:
        image = self.render(
            image=image,
            spec=spec,
            mask_range=project.ggx.mask.toTuple(),
            amount=project.ggx.amount,
        )
        return image

import logging
from functools import lru_cache

import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project
from autochrome.api.tasks.opencl import OpenCL, Image
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)


class GGXTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
        self.kernel = None
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = self.read_source_file('ggx.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'BRDF')

    @lru_cache(1)
    def render(
        self,
        resolution: QtCore.QSize,
        roughness: float,
        height: float,
    ) -> Image:
        if self.rebuild:
            self.build()

        # create output buffer
        ggx = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)
        ggx.args = (resolution, roughness, height)

        # run program
        self.kernel.set_arg(0, ggx.image)
        self.kernel.set_arg(1, np.float32(roughness))
        self.kernel.set_arg(2, np.float32(height))

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )
        cl.enqueue_copy(self.queue, ggx.array, ggx.image, origin=(0, 0), region=(w, h))

        return ggx

    @timer
    def run(self, project: Project) -> Image:
        image = self.render(
            resolution=project.halation.resolution,
            roughness=project.halation.roughness,
            height=project.halation.height,
        )
        return image

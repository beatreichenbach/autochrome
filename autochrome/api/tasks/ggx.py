import logging

import numpy as np
import pyopencl as cl
from PySide2 import QtCore

from autochrome.api.data import Project
from autochrome.api.tasks.opencl import OpenCL, Image

logger = logging.getLogger(__name__)


class GGXTask(OpenCL):
    def __init__(self, queue) -> None:
        super().__init__(queue)
        self.kernel = None
        self.build()

    def build(self, *args, **kwargs) -> None:
        self.source = ''
        self.source += self.read_source_file('ggx.cl')

        super().build()

        self.kernel = cl.Kernel(self.program, 'BRDF')

    def render(
        self,
        resolution: QtCore.QSize,
        roughness: float,
        height: float,
        light_position: tuple[float, float],
    ) -> Image:
        if self.rebuild:
            self.build()

        # create output buffer
        ggx = self.update_image(resolution, flags=cl.mem_flags.READ_WRITE)
        ggx.args = (
            resolution,
            roughness,
            height,
            light_position,
        )

        light_position_cl = np.array(light_position, dtype=cl.cltypes.float2)

        # run program
        self.kernel.set_arg(0, ggx.image)
        self.kernel.set_arg(1, np.float32(roughness))
        self.kernel.set_arg(2, np.float32(height))
        self.kernel.set_arg(3, light_position_cl)

        w, h = resolution.width(), resolution.height()
        global_work_size = (w, h)
        local_work_size = None

        cl.enqueue_nd_range_kernel(
            self.queue, self.kernel, global_work_size, local_work_size
        )
        cl.enqueue_copy(self.queue, ggx.array, ggx.image, origin=(0, 0), region=(w, h))

        return ggx

    def run(self, project: Project) -> Image:
        light_position = (
            project.halation.light_position.x(),
            project.halation.light_position.y(),
        )
        image = self.render(
            resolution=project.halation.resolution,
            roughness=project.halation.roughness,
            height=project.halation.height,
            light_position=light_position,
        )
        return image

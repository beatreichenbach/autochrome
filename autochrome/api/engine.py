from __future__ import annotations

import logging
import os
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore
from pyopencl import tools

from autochrome.api.data import Project, RenderElement, RenderImage, EngineError
from autochrome.api.path import File
from autochrome.api.tasks import opencl
from autochrome.api.tasks.emulsion import EmulsionTask
from autochrome.api.tasks.ggx import GGXTask
from autochrome.api.tasks.grain import GrainTask
from autochrome.api.tasks.halation import HalationTask
from autochrome.api.tasks.jakob import SpectralTask
from autochrome.api.tasks.opencl import Image
from autochrome.storage import Storage
from autochrome.utils import ocio
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)
storage = Storage()


@lru_cache(3)
def apply_colorspace(image: Image, src_name: str) -> Image:
    array = image.array.copy()
    processor = ocio.colorspace_processor(src_name=src_name)
    if processor:
        processor.applyRGBA(array)
    image = Image(image.context, array=array, args=(image, src_name))
    return image


class Engine(QtCore.QObject):
    image_rendered: QtCore.Signal = QtCore.Signal(RenderImage)
    progress_changed: QtCore.Signal = QtCore.Signal(float)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)

        self._emit_cache = {}
        self._elements = []

        self.queue = None

    def init(self, device: str = '') -> None:
        """Initializes the engine. This needs to happen in a different function to
        create all objects in the right thread."""
        self.queue = opencl.command_queue(device)
        logger.debug(f'Engine initialized on device: {self.queue.device.name}')
        self._init_renderers()
        self._init_tasks()

    def _init_renderers(self) -> None:
        self.renderers = {
            RenderElement.INPUT: self.source,
            RenderElement.EMULSION_R: self.emulsion,
            RenderElement.EMULSION_G: self.emulsion,
            RenderElement.EMULSION_B: self.emulsion,
            RenderElement.KERNEL: self.ggx,
            RenderElement.HALATION: self.halation,
            RenderElement.GRAIN: self.grain,
        }

    def _init_tasks(self) -> None:
        self.spectral_task = SpectralTask()
        self.emulsion_task = EmulsionTask(self.queue)
        self.ggx_task = GGXTask(self.queue)
        self.halation_task = HalationTask(self.queue)
        self.grain_task = GrainTask(self.queue)

    def elements(self) -> list[RenderElement]:
        return self._elements

    def source(self, project: Project) -> Image:
        image_file = File(project.input.image_path)
        resolution = (
            project.render.resolution if project.render.force_resolution else None
        )
        image = self.emulsion_task.load_file(image_file, resolution)
        image = apply_colorspace(image, project.input.colorspace)
        return image

    def emulsion(self, project: Project) -> Image:
        self.spectral_task.run(project)
        spectral_images = self.emulsion_task.run(project)

        # TODO: colorspace should be stored in Image
        element = next(
            (e for e in self._elements if e.name.startswith('EMULSION')),
            RenderElement.EMULSION_R,
        )

        if element == RenderElement.EMULSION_B:
            emulsion = spectral_images[2]
        elif element == RenderElement.EMULSION_G:
            emulsion = spectral_images[1]
        else:
            emulsion = spectral_images[0]

        # src_name = 'CIE-XYZ-D65'
        src_name = 'Utility - XYZ - D60'
        image = apply_colorspace(emulsion, src_name)

        return image

    def ggx(self, project: Project) -> Image:
        image = self.ggx_task.run(project)
        return image

    def halation(self, project: Project) -> Image:
        self.spectral_task.run(project)
        spectral_images = self.emulsion_task.run(project)
        kernel = self.ggx_task.run(project)
        halation = self.halation_task.run(project, spectral_images[0], kernel)

        # TODO: colorspace should be stored in Image
        # src_name = 'CIE-XYZ-D65'
        src_name = 'Utility - XYZ - D60'
        image = apply_colorspace(halation, src_name)

        return image

    def grain(self, project: Project) -> Image:
        self.spectral_task.run(project)
        spectral_images = self.emulsion_task.run(project)
        kernel = self.ggx_task.run(project)
        halation = self.halation_task.run(project, spectral_images[0], kernel)
        spectral_images = (halation, spectral_images[1], spectral_images[2])

        grain = self.grain_task.run(project, spectral_images)

        # TODO: colorspace should be stored in Image
        src_name = 'Output - sRGB'
        image = apply_colorspace(grain, src_name)

        # src_name = 'CIE-XYZ-D65'
        src_name = 'Utility - XYZ - D60'
        image = apply_colorspace(image, src_name)

        return image

    @timer
    def render(self, project: Project) -> bool:
        if not self.queue:
            self.init()

        self.progress_changed.emit(0)
        try:
            for element in self._elements:
                renderer = self.renderers.get(element)
                if renderer:
                    image = renderer(project)
                    self.emit_image(image, element)
                    self.write_image(image, element, project)
        except EngineError as e:
            logger.error(e)
        except cl.Error as e:
            logger.exception(e)
            logger.error(
                'Render failed. This is most likely because the GPU ran out of memory. '
                'Consider lowering the settings and restarting the engine.'
            )
        except InterruptedError:
            logger.warning('Render interrupted by user')
            return False
        except Exception as e:
            logger.exception(e)
            return False
        finally:
            self.progress_changed.emit(1)
        return True

    def set_elements(self, elements: list[RenderElement]) -> None:
        self._elements = elements
        # clear cache to force updates to viewers
        self._emit_cache = {}

    def emit_image(self, image: Image, element: RenderElement) -> None:
        # NOTE: Emits image_rendered signal if hash for that element has changed
        # one hash per element is stored, so lru_cache is not used here
        _hash = hash(image)
        if self._emit_cache.get(element) != _hash:
            self._emit_cache[element] = _hash
            render_image = RenderImage(image, element)
            self.image_rendered.emit(render_image)

    def write_image(
        self, image: Image, element: RenderElement, project: Project
    ) -> None:
        if not project.output.write or element != project.output.element:
            return
        filename = storage.parse_output_path(project.output.path, project.output.frame)
        write_array(image.array, filename, project.output.colorspace)


def clear_cache() -> None:
    cl.tools.clear_first_arg_caches()


def write_array(array: np.ndarray, filename: str, colorspace: str) -> None:
    array = array.copy()

    # colorspace
    processor = ocio.colorspace_processor(dst_name=colorspace)
    if processor:
        processor.applyRGBA(array)

    # 8bit
    if not filename.endswith('.exr'):
        array *= 255

    try:
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        image_bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(filename, image_bgr)
        logger.info('image written: {}'.format(filename))
    except (OSError, ValueError, cv2.error) as e:
        logger.debug(e)
        message = f'Error writing file: {filename}'
        raise EngineError(message) from None

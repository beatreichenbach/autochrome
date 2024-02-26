from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Callable

import cv2
import numpy as np
import pyopencl as cl
from PySide2 import QtCore
from pyopencl import tools

from autochrome.api.data import Project, RenderElement, RenderImage, EngineError
from autochrome.api.tasks import opencl
from autochrome.api.tasks.grain import GrainTask
from autochrome.api.tasks.opencl import Image
from autochrome.api.tasks.spectral import SpectralTask
from autochrome.storage import Storage

from autochrome.utils import ocio
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)
storage = Storage()


class Engine(QtCore.QObject):
    image_rendered: QtCore.Signal = QtCore.Signal(RenderImage)
    progress_changed: QtCore.Signal = QtCore.Signal(float)

    def __init__(self, device: str = '', parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)

        self.queue = opencl.command_queue(device)
        logger.debug(f'engine initialized on device: {self.queue.device.name}')

        self._emit_cache = {}
        self._elements = []
        self._init_renderers()
        self._init_tasks()

    def _init_renderers(self) -> None:
        self.renderers: dict[RenderElement, Callable] = OrderedDict()
        self.renderers[RenderElement.SPECTRAL] = self.spectral
        self.renderers[RenderElement.GRAIN] = self.grain

    def _init_tasks(self) -> None:
        self.grain_task = GrainTask(self.queue)
        self.spectral_task = SpectralTask(self.queue)

    def elements(self) -> list[RenderElement]:
        return self._elements

    def spectral(self, project: Project) -> Image:
        image = self.spectral_task.run(project)
        return image

    def grain(self, project: Project) -> Image:
        spectral_buffer = self.spectral_task.run_buffer(project)
        image = self.grain_task.run(project)
        return image

    @timer
    def render(self, project: Project) -> bool:
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
        # emits image_rendered signal if hash for that element has changed
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

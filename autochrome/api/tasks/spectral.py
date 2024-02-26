import enum
import itertools
import logging
import os.path
from functools import lru_cache

import cv2
import numpy as np
import torch
from PySide2 import QtCore
from torch import nn

from autochrome.api.data import Project, EngineError
from autochrome.api.path import File
from autochrome.api.tasks.mst_plus_plus import MST_Plus_Plus
from autochrome.api.tasks.opencl import OpenCL, Image, Buffer
from autochrome.utils import ocio
from autochrome.utils.ciexyz import CIEXYZ
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)


class Center(enum.Enum):
    median = enum.auto()
    mean = enum.auto()


def _transform(
    data: torch.Tensor,
    flip_x: bool,
    flip_y: bool,
    transpose: bool,
    reverse: bool = False,
) -> torch.Tensor:
    if not reverse:
        if flip_x:
            data = torch.flip(data, [3])
        if flip_y:
            data = torch.flip(data, [2])
        if transpose:
            data = torch.transpose(data, 2, 3)
    else:
        if transpose:
            data = torch.transpose(data, 2, 3)
        if flip_y:
            data = torch.flip(data, [2])
        if flip_x:
            data = torch.flip(data, [3])
    return data


def _forward_ensemble(
    tensor: torch.Tensor, model: nn.Module, center: Center = Center.mean
) -> torch.Tensor:
    outputs = []
    options = itertools.product((False, True), (False, True), (False, True))
    for flip_x, flip_y, transpose in options:
        data = tensor.clone()
        data = _transform(data, flip_x, flip_y, transpose)
        data = model(data)
        data = _transform(data, flip_x, flip_y, transpose, reverse=True)
        outputs.append(data)

    if center == Center.mean:
        return torch.stack(outputs, 0).mean(0)
    elif center == Center.median:
        return torch.stack(outputs, 0).median(0)[0]
    else:
        ValueError(f'Unsupported center: {center}')


class SpectralTask(OpenCL):
    @lru_cache(1)
    def load_file(self, file: File, resolution: QtCore.QSize) -> Image:
        # load array
        filename = str(file)
        try:
            array = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        except ValueError as e:
            logger.debug(e)
            message = f'Invalid Image path: {filename}'
            raise EngineError(message) from None

        # convert to float32
        if array.dtype == np.uint8:
            array = np.divide(array, 255)
        array = np.float32(array)

        # resize array
        array = cv2.resize(array, (resolution.width(), resolution.height()))

        # return image
        image = Image(self.context, array=array, args=(file, resolution))
        return image

    @lru_cache(1)
    def update_model(self, model_path: str | None = None) -> nn.Module:
        model = MST_Plus_Plus().cuda()
        if model_path is not None:
            logger.info(f'Loading model from: {model_path}')
            checkpoint = torch.load(model_path)
            state_dict = {
                k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()
            }
            model.load_state_dict(state_dict, strict=True)
        return model.cuda()

    @timer
    def update_spectral_array(
        self, image: Image, model: nn.Module, ensemble_center: Center.mean
    ) -> np.ndarray:
        rgb = image.array
        rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb_aligned = np.expand_dims(np.transpose(rgb_normalized, [2, 0, 1]), axis=0)

        torch_rgb = torch.from_numpy(rgb_aligned).float().cuda()
        with torch.no_grad():
            torch_result = _forward_ensemble(
                tensor=torch_rgb, model=model, center=ensemble_center
            )

        spectral_array = torch_result.cpu().numpy()
        spectral_array = np.transpose(np.squeeze(spectral_array), [1, 2, 0])
        # spectral_array = np.clip(spectral_array, 0, 1)

        return spectral_array

    @timer
    def update_preview(self, spectral_array: np.ndarray) -> Image:
        height, width, lambda_count = spectral_array.shape
        logger.debug(f'lambda_count: {lambda_count}')
        rgb_dict = {w: np.float32((r, g, b)) for (w, r, g, b) in CIEXYZ}
        lambda_min = 400
        lambda_max = 700
        lambda_values = np.linspace(lambda_min, lambda_max, lambda_count)
        image_shape = (height, width, 3)
        output = np.zeros(image_shape, np.float32)
        for w in range(lambda_count):
            rgb = rgb_dict[lambda_values[w]]
            for y in range(height):
                for x in range(width):
                    output[y, x] += rgb * spectral_array[y, x, w]

        # return image
        image = Image(self.context, array=output)
        return image

    @timer
    @lru_cache(1)
    def spectral_buffer(
        self,
        image_file: File,
        resolution: QtCore.QSize,
        model_path: str | None,
        ensemble_center: Center.mean,
    ) -> Buffer:
        image = self.load_file(image_file, resolution)
        model = self.update_model(model_path)
        array = self.update_spectral_array(image, model, ensemble_center)
        spectral = Buffer(
            self.context, array=array, args=(image, model, ensemble_center)
        )
        return spectral

    @timer
    @lru_cache(1)
    def spectral_image(
        self,
        image_file: File,
        resolution: QtCore.QSize,
        model_path: str | None,
        ensemble_center: Center,
    ) -> Image:
        image = self.load_file(image_file, resolution)

        image_min = image.array.min()
        image_delta = image.array.max() - image.array.min()
        logger.debug(f'image_min: {image_min}')
        logger.debug(f'image_delta: {image_delta}')

        model = self.update_model(model_path)
        spectral_array = self.update_spectral_array(image, model, ensemble_center)
        image = self.update_preview(spectral_array)
        processor = ocio.colorspace_processor(src_name='CIE-XYZ-D65')

        # processor = ocio.colorspace_processor(src_name='sRGB - Display')

        array = image.array * image_delta - image_min
        array *= 1 / 2**4
        # add alpha channel
        array = np.dstack((array, np.zeros(array.shape[:2], np.float32)))
        if processor:
            processor.applyRGBA(array)
        image._array = array

        return image

    def run_buffer(self, project: Project) -> Buffer:
        image_file = File(project.input.image_path)
        resolution = project.render.resolution
        model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'resources',
            'models',
            'mst_plus_plus.pth',
        )
        ensemble_center = Center.mean
        buffer = self.spectral_buffer(
            image_file, resolution, model_path, ensemble_center
        )
        return buffer

    def run(self, project: Project) -> Image:
        image_file = File(project.input.image_path)
        resolution = project.render.resolution
        model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'resources',
            'models',
            'mst_plus_plus.pth',
        )
        ensemble_center = Center.mean
        image = self.spectral_image(image_file, resolution, model_path, ensemble_center)
        return image

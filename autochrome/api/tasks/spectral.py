import logging
from functools import lru_cache

import cv2
import numpy as np
import pyopencl as cl
import colour
from PySide2 import QtCore
from colour.colorimetry import sd_to_XYZ_integration

from autochrome.api.path import File
from autochrome.api.data import Project, EngineError
from autochrome.api.tasks.opencl import OpenCL, Image, LAMBDA_MIN, LAMBDA_MAX

logger = logging.getLogger(__name__)

# https://colour.readthedocs.io/en/master/generated/colour.XYZ_to_sd.html


class SpectralTask(OpenCL):
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

    @lru_cache(10)
    def update_cmfs(self, interval: int = 20) -> colour.SpectralShape:
        spectral_shape = colour.SpectralShape(LAMBDA_MIN, LAMBDA_MAX, interval)
        logger.debug(f'wavelengths: {spectral_shape.wavelengths}')
        cmfs = (
            colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
            .copy()
            .align(spectral_shape)
        )
        return cmfs

    def update_spectral_distribution(self, image_array: np.ndarray) -> np.ndarray:

        cmfs = self.update_cmfs()
        illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        shape = (image_array.shape[0], image_array.shape[1], len(cmfs.shape))
        array = np.zeros(shape, dtype=np.float32)
        logger.debug(array.shape)
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                xyz = image_array[i, j]
                sd = colour.XYZ_to_sd(
                    xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant
                )
                array[i, j] = sd.values
        return array

    def update_preview(self, array: np.array, wavelength: int) -> Image:
        cmfs = self.update_cmfs()
        absolute_diff = np.abs(cmfs.wavelengths - wavelength)
        index = np.argmin(absolute_diff)
        logger.debug(index)
        array = array[:, :, index]
        image = Image(self.context, array=array, args=wavelength)
        return image

    def run(self, project: Project) -> Image:
        image_file = File(project.input.image_path)
        resolution = project.render.resolution
        image_array = self.load_file(image_file, resolution)
        # spectral_distribution = self.update_spectral_distribution(image_array)
        # spectral = self.update_preview(
        #     spectral_distribution, project.spectral.wavelength
        # )
        image = Image(self.context, image_array)
        return image

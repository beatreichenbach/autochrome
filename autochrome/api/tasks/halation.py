import logging
from functools import lru_cache

import numpy as np

from autochrome.api.data import Project
from autochrome.api.tasks.opencl import OpenCL, Image
from autochrome.utils.timing import timer

logger = logging.getLogger(__name__)


def fft_convolve2d(
    source: np.ndarray, kernel: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """Return a 2D convolution."""
    fft_source = np.fft.fft2(source)
    # TODO: handle both dimensions
    padding = (np.array(source.shape) - np.array(kernel.shape)) / 2
    pad = int(padding[0]), int(padding[1])
    kernel = np.pad(kernel, ((pad[0],), (pad[1],)), mode='constant')

    # NOTE: To center the kernel, flip it upside down and left right.
    #       To normalize the kernel use norm='ortho' attribute
    norm = 'ortho' if normalize else 'backward'
    fft_kernel = np.fft.fft2(np.flipud(np.fliplr(kernel)), norm=norm)

    convolved = np.real(np.fft.ifft2(fft_source * fft_kernel))
    convolved = np.fft.ifftshift(convolved)

    return convolved


class HalationTask(OpenCL):
    @lru_cache(1)
    def render(
        self,
        image: Image,
        kernel: Image,
        threshold: float,
        amount: float,
        mask_only: bool,
        halation_only: bool,
    ):
        args = (image, kernel, threshold, amount, mask_only, halation_only)
        luminance = image.array[:, :, 1]

        mask = np.where(luminance > threshold, luminance - threshold, 0)
        mask = mask[:, :, np.newaxis]

        if mask_only:
            output = Image(self.queue, array=mask, args=args)
            return output

        masked_array = image.array * mask

        # TODO: padding
        pad = kernel.array.shape[0]
        halation = np.zeros(image.array.shape, np.float32)
        for i in range(3):
            logger.debug(f'Rendering convolution kernel for layer {i} ...')
            padded_array = np.pad(masked_array[:, :, i], pad, mode='constant')
            logger.debug(f'{padded_array.shape=}')
            convolve = fft_convolve2d(padded_array, kernel.array[:, :, 0], True)
            logger.debug(f'{convolve.shape=}')
            halation[:, :, i] = convolve[pad:-pad, pad:-pad]

        if halation_only:
            output = Image(self.queue, array=halation, args=args)
            return output

        logger.debug(f'{halation.shape=}')

        output = Image(self.queue, array=image.array + halation * amount, args=args)
        return output

    @timer
    def run(self, project: Project, image: Image, kernel: Image) -> Image:
        image = self.render(
            image=image,
            kernel=kernel,
            threshold=project.halation.threshold,
            amount=project.halation.amount,
            mask_only=project.halation.mask_only,
            halation_only=project.halation.halation_only,
        )
        return image

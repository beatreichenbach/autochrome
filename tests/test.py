import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(filename: str) -> np.ndarray:
    # load array
    try:
        array = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    except ValueError as e:
        logger.debug(e)
        message = f'Invalid Image path: {filename}'
        raise ValueError(message) from None

    # convert to float32
    if array.dtype == np.uint8:
        array = np.divide(array, 255)
    array = np.float32(array)

    # resize array
    array = cv2.resize(array, (512, 512))
    return array

import logging
import sys

import numpy as np
from PySide2 import QtWidgets

from autochrome.api.data import Project
from autochrome.api.tasks import opencl
from autochrome.api.tasks.halation import fft_convolve2d, HalationTask
from autochrome.api.tasks.opencl import Image
from qt_extensions.viewer import Viewer


def scale(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rows, columns = image.shape
    scaled = [
        [image[int(rows * r / height)][int(columns * c / width)] for c in range(width)]
        for r in range(height)
    ]
    return np.array(scaled)


def get_checkerboard():
    checkerboard = np.float32(np.indices((16, 16)).sum(axis=0) % 2)

    checkerboard_scaled = scale(checkerboard, 512, 512)

    ramp = np.tile(
        np.linspace(start=0, stop=1, num=512, dtype=np.float32), reps=(512, 1)
    )
    checkerboard_scaled *= ramp
    return checkerboard_scaled


def get_kernel():
    x_axis = np.linspace(-1, 1, 64)[:, None]
    y_axis = np.linspace(-1, 1, 64)[None, :]
    kernel = np.clip(-np.sqrt(x_axis**2 + y_axis**2) + 1, 0, 1)
    return kernel


def test_convolve2d():
    checkerboard = get_checkerboard()
    kernel = get_kernel()

    pad = 32
    checkerboard = np.pad(checkerboard, pad, mode='constant')
    output = fft_convolve2d(checkerboard, kernel)
    output = output[pad:-pad, pad:-pad]

    app = QtWidgets.QApplication(sys.argv)

    viewer = Viewer()
    viewer.set_array(output)
    viewer.show()

    sys.exit(app.exec_())


def test_halation():
    queue = opencl.command_queue()
    halation_task = HalationTask(queue)
    checkerboard = get_checkerboard()
    checkerboard = np.dstack((checkerboard,) * 3)
    print(checkerboard.shape)

    image = Image(queue, array=checkerboard)
    kernel = Image(queue, array=get_kernel())
    project = Project()
    project.halation.mask_only = False
    halation_image = halation_task.run(project=project, image=image, kernel=kernel)

    app = QtWidgets.QApplication(sys.argv)

    viewer = Viewer()
    viewer.set_array(halation_image.array)
    viewer.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_halation()

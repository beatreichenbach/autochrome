import logging


from autochrome.api.data import Project
from autochrome.api.tasks import opencl
from autochrome.api.tasks.emulsion import EmulsionTask
from autochrome.api.tasks.jakob import fetch, eval_precise, get_cmfs, get_illuminant
from autochrome.utils import ocio

import logging
import sys

import numpy as np
from PySide2 import QtWidgets, QtGui

from qt_extensions import theme
from qt_extensions.viewer import Viewer

SRGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

XYZ_TO_SRGB = np.array(
    [
        [3.240479, -1.537150, -0.498535],
        [-0.969256, 1.875991, 0.041556],
        [0.055648, -0.204043, 1.057311],
    ]
)


def test_pattern(resolution: int):
    space = np.linspace(0, 1, resolution)
    pattern_3d = np.stack(np.meshgrid(space, space, space), axis=3)

    shape = (resolution, resolution**2, 3)
    pattern_2d = pattern_3d.flatten()
    pattern_2d.resize(shape, refcheck=False)
    return pattern_2d


def reconstruct(array: np.ndarray) -> np.ndarray:
    model = np.load(
        r'/home/beat/dev/autochrome/autochrome/api/tasks/afef8e3dd5781a7df338b7f65ffcc0ad.npy'
    )
    lambda_min = 390
    lambda_max = 830
    lambda_count = 21
    resolution = 16
    reconstructed = np.zeros(array.shape)
    lambdas = np.linspace(lambda_min, lambda_max, lambda_count)
    standard_illuminant = 'D65'
    cmfs_variation = 'CIE 2015 2 Degree Standard Observer'
    cmfs = get_cmfs(cmfs_variation, lambdas)
    illuminant = get_illuminant(standard_illuminant, lambdas)

    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            rgb = array[y, x]
            xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)

            coefficients = fetch(model, resolution, xyz)
            sd = np.array(
                [
                    eval_precise(coefficients, l / (lambda_count - 1))
                    for l in range(lambda_count)
                ]
            )

            xyz = np.dot(sd * illuminant, cmfs) / k
            rgb = np.clip(np.dot(XYZ_TO_SRGB, xyz), 0, 1)
            reconstructed[y, x] = rgb
    return reconstructed


def test_emulsion_task():
    queue = opencl.command_queue()
    spectral_task = EmulsionTask(queue)
    spectral_images = spectral_task.run(project=Project())

    output = (
        spectral_images[0].array + spectral_images[1].array + spectral_images[2].array
    )

    # pattern = test_pattern(5)
    # pattern_reconstructed = reconstruct(pattern)
    # diff = np.abs(pattern - pattern_reconstructed)
    # output = np.vstack((pattern, pattern_reconstructed, diff))

    app = QtWidgets.QApplication(sys.argv)

    viewer = Viewer()
    viewer.set_array(output)
    viewer.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_emulsion_task()

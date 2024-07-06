import logging

import numpy as np
from numpy.testing import assert_array_equal

from autochrome.api.data import Project
from autochrome.api.tasks import jakob


def test_decomposition():
    a = np.array([[1, 4, 5], [6, 8, 22], [32, 5, 5]], np.double)
    b = np.array([1, 2, 3], np.double)

    tolerance = 1e-15
    a, permutation = jakob.decompose(a, tolerance)
    x = jakob.solve(a, permutation, b)
    logging.info(x)

    expected_x = np.linalg.solve(a, b)
    logging.info(expected_x)

    # expected_x = np.array([0, 0.5, 0], np.float16)
    # assert_array_equal(x, expected_x)

    x = jakob.lu_solve(a, b)
    logging.info(x)


def test_spectral_task():
    spectral_task = jakob.SpectralTask()
    spectral_task.run(project=Project())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_decomposition()

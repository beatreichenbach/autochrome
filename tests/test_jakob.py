import logging

import numpy as np
from numpy.testing import assert_array_equal

from autochrome.api.data import Project
from autochrome.api.tasks.jakob import decompose, solve, SpectralTask


def test_decomposition():
    a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], np.float16)
    tolerance = 1e-15
    a, permutation = decompose(a, tolerance)
    b = np.array([1, 2, 3], np.float16)
    x = solve(a, permutation, b)
    expected_x = np.array([0, 0.5, 0], np.float16)

    assert_array_equal(x, expected_x)


def test_spectral_task():
    spectral_task = SpectralTask()
    spectral_task.run(project=Project())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_spectral_task()

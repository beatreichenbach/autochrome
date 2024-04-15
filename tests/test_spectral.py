import logging
import sys

import numpy as np
from PySide2 import QtWidgets
import colour

from autochrome.api.tasks.conversion import (
    fetch,
    eval_precise,
    model_path,
    lambdas,
    cmfs,
    illuminant,
    LAMBDA_MIN,
    LAMBDA_MAX,
    gauss_newton,
    LAMBDA_COUNT,
    smoothstep,
)
from qt_extensions.viewer import Viewer

logger = logging.getLogger(__name__)


def update_test_pattern(resolution: int):
    space = np.linspace(0, 1, resolution)
    pattern_3d = np.stack(np.meshgrid(space, space, space), axis=3)

    # size = int(np.ceil(np.sqrt(resolution**3)))
    shape = (resolution, resolution**2, 3)
    pattern_2d = pattern_3d.flatten()
    pattern_2d.resize(shape, refcheck=False)
    return pattern_2d


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


def create_model_3d(model_path: str, resolution: int):
    space = np.linspace(0, 1, resolution)
    grid = np.stack(np.meshgrid(space, space, space), axis=3)
    # scale = [smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)]

    count = resolution * resolution - 1
    model = np.ndarray((resolution, resolution, resolution, 3))
    for x in range(resolution):
        for y in range(resolution):
            logger.info(f'{((x * resolution) + y) / count * 100:.01f} %')
            for z in range(resolution):
                # NOTE: not sure why y, x, z
                xyz = grid[y, x, z]
                # xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)
                coefficients = gauss_newton(xyz, np.zeros(3))
                model[x, y, z] = coefficients

    np.save(model_path, model)


def trilinear_interpolation(xyz: np.ndarray, model: np.ndarray):
    x, y, z = xyz
    model_resolution = model.shape[0]
    xyz0 = np.minimum(np.int32(xyz * (model_resolution - 1)), model_resolution - 2)

    xd = x * (model_resolution - 1) - xyz0[0]
    yd = y * (model_resolution - 1) - xyz0[1]
    zd = z * (model_resolution - 1) - xyz0[2]

    c000 = model[tuple(xyz0)]
    c100 = model[tuple(xyz0 + np.array((1, 0, 0)))]
    c010 = model[tuple(xyz0 + np.array((0, 1, 0)))]
    c001 = model[tuple(xyz0 + np.array((0, 0, 1)))]
    c110 = model[tuple(xyz0 + np.array((1, 1, 0)))]
    c101 = model[tuple(xyz0 + np.array((1, 0, 1)))]
    c011 = model[tuple(xyz0 + np.array((0, 1, 1)))]
    c111 = model[tuple(xyz0 + np.array((1, 1, 1)))]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    model_resolution = 32
    model_path_3d = '/home/beat/dev/autochrome/autochrome/api/tasks/model_3d.npy'
    # create_model_3d(model_path, model_resolution)
    model = np.load(model_path_3d)

    pattern_resolution = 10
    pattern = update_test_pattern(pattern_resolution)
    pattern_reconstructed = np.zeros(pattern.shape)

    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)

    # cmfs_cie_2 = colour.MSDS_CMFS['CIE 2015 2 Degree Standard Observer']
    # d_w = 5
    # spectral_shape = colour.SpectralShape(LAMBDA_MIN, LAMBDA_MAX, d_w)
    # cmfs = cmfs_cie_2.copy().align(spectral_shape)
    # illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)

    for p_y in range(pattern.shape[0]):
        for p_x in range(pattern.shape[1]):
            logger.info((p_y, p_x))

            rgb = pattern[p_y, p_x]
            xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)
            # xyz = colour.sRGB_to_XYZ(rgb)

            # pattern[p_y, p_x] = xyz
            # logger.debug(xyz)

            # -- interpolate --
            # xyz = pattern[p_y, p_x]
            # coefficients = trilinear_interpolation(xyz, model)
            # sd = np.array(
            #     [
            #         eval_precise(coefficients, l / (LAMBDA_COUNT - 1))
            #         for l in range(LAMBDA_COUNT)
            #     ]
            # )

            # -- fetch --
            coefficients = fetch(xyz, model_path)
            # sd = np.array([eval_precise(coefficients, l) for l in lambdas])

            # -- colour --
            # sd = colour.XYZ_to_sd(
            #     xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant
            # )
            # xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100

            # -- solve --
            # coefficients = gauss_newton(xyz, np.zeros(3))
            sd = np.array(
                [
                    eval_precise(coefficients, l / (LAMBDA_COUNT - 1))
                    for l in range(LAMBDA_COUNT)
                ]
            )

            # pattern_reconstructed[p_y, p_x] = coefficients
            # continue

            xyz = np.dot(sd * illuminant, cmfs) / k
            rgb = np.clip(np.dot(XYZ_TO_SRGB, xyz), 0, 1)
            pattern_reconstructed[p_y, p_x] = rgb

    diff = np.abs(pattern - pattern_reconstructed)
    output = np.vstack((pattern, pattern_reconstructed, diff))

    logger.info(np.mean(diff))

    app = QtWidgets.QApplication(sys.argv)
    viewer = Viewer()
    viewer.set_array(output)
    viewer.show()
    sys.exit(app.exec_())

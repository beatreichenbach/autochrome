# https://xavierbourretsicotte.github.io/coordinate_descent.html

import logging
from functools import lru_cache
import colour
from colour.colorimetry import sd_to_XYZ_integration

import numpy as np

from autochrome.utils.ciexyz import CIEXYZ

logger = logging.getLogger(__name__)

LAMBDA_MIN = 400
LAMBDA_MAX = 700
LAMBDA_COUNT = 15


@lru_cache
def rgb_curve(lambda_min: int, lambda_max: int, lambda_count: int) -> np.ndarray:
    lambda_values = np.linspace(lambda_min, lambda_max, lambda_count, dtype=np.int32)
    logger.info(lambda_values)
    # rgb_dict = {w: np.float32((r, g, b)) for (w, r, g, b) in CIEXYZ}
    # array = np.ndarray((lambda_count, 3), np.float32)
    # for w in range(lambda_count):
    #    lambda_value = int(lambda_values[w])
    #    array[w] = rgb_dict[lambda_value]
    array = np.float32([[r, g, b] for (w, r, g, b) in CIEXYZ if w in lambda_values])
    return array


def color_distance(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    # https://facelessuser.github.io/coloraide/distance
    distance = np.linalg.norm(rgb1 - rgb2)
    return distance


def spectrum_to_rgb(spectrum: np.ndarray) -> np.ndarray:
    curve = rgb_curve(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT + 1)
    output = curve * spectrum[:, np.newaxis]
    rgb = np.sum(output, axis=0)

    return rgb


def spectrum_to_xyz(spectrum: np.ndarray) -> np.ndarray:
    data = {}
    for k in range(spectrum.shape[0]):
        w = cmfs.wavelengths[k]
        data[w] = spectrum[k]
    sd_ = colour.SpectralDistribution(data, cmfs=cmfs, illuminant=illuminant)
    xyz_ = colour.sd_to_XYZ(sd_, cmfs=cmfs, illuminant=illuminant)
    return xyz_


# def convert():
#     spectrum_initial = np.zeros((LAMBDA_COUNT,), np.float32)
#     coordinate_descent(spectrum_initial, max_iterations=10, tolerance=0.1)

# logger.info(f'rgb: {rgb}')


# def cost(spectrum: np.ndarray, result: np.ndarray, spectrum: np.ndarray) -> np.ndarray:


# def costfunction(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
#     m = np.size(y)
#     h = np.matmul(X, theta)
#     c = 1 / (2 * m) * np.matmul((h - y).T, (h - y))
#     # rgb = spectrum_to_rgb(spectrum)
#     # color_distance(rgb, result)
#     return c


# def coordinate_descent(theta, X, y, alpha=0.03, max_iterations=20):
#     # theta: the initial values for the parameters (in this case the initial spectrum)
#     # X:
#     m, n = X.shape
#     cost_history = []
#     theta_0_hist, theta_1_hist = [], []  # For plotting afterwards
#
#     for i in range(max_iterations):
#
#         for j in range(n):
#             # Coordinate descent in vectorized form
#             h = np.matmul(X, theta)
#             gradient = np.matmul(X[:, j], (h - y))
#             theta[j] = theta[j] - alpha * gradient
#
#             # Saving values for plots
#             cost_history.append(costfunction(X, y, theta))
#             theta_0_hist.append(theta[0, 0])
#             theta_1_hist.append(theta[1, 0])
#
#     return theta, cost_history, theta_0_hist, theta_1_hist


def diff(spectrum: np.ndarray, target: np.ndarray) -> float:
    xyz_ = spectrum_to_xyz(spectrum) / 100
    distance = color_distance(xyz_, target)
    return distance


def coordinate_descent(
    spectrum: np.ndarray, target: np.ndarray, max_iterations: int, tolerance: float
) -> np.ndarray:
    lambda_count = spectrum.shape[0]
    steps = np.ones((lambda_count,)) * 0.01
    for k in range(max_iterations):
        for i in range(lambda_count):
            diff_initial = diff(spectrum, target)

            spectrum[i] += steps[i]
            if spectrum[i] < 0:
                spectrum[i] = 0
                steps[i] *= -1
                break

            new_diff = diff(spectrum, target)
            if new_diff <= tolerance:
                logger.info(f'early termination after {k} iterations')
                return spectrum
            if new_diff > diff_initial:
                steps[i] *= -0.5
                break
    return spectrum


@lru_cache(10)
def update_cmfs() -> colour.SpectralShape:
    interval = (LAMBDA_MAX - LAMBDA_MIN) / LAMBDA_COUNT
    spectral_shape = colour.SpectralShape(LAMBDA_MIN, LAMBDA_MAX, interval)
    logger.debug(f'wavelengths_count: {len(spectral_shape.wavelengths)}')
    cmfs = (
        colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy()
        .align(spectral_shape)
    )
    logger.debug(f'cmfs_shape: {cmfs.shape}')
    return cmfs


def spectral_distribution(xyz: np.ndarray) -> np.ndarray:
    cmfs = update_cmfs()
    illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    sd = colour.XYZ_to_sd(xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
    return sd.values


# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#
#     lambda_values = np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT + 1)
#     cmfs = update_cmfs()
#     illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
#
#     srgb = np.array([0.95, 0.8, 0.1], np.float32)
#     xyz = colour.sRGB_to_XYZ(srgb)
#     logger.info(f'xyz: {xyz}')
#
#     # custom
#     spectrum_initial = np.zeros((LAMBDA_COUNT + 1,), np.float32)
#     spectrum_approx = coordinate_descent(
#         spectrum_initial, xyz, max_iterations=100, tolerance=0.001
#     )
#     custom_xyz_custom = spectrum_to_rgb(spectrum_approx)
#     logger.info(f'custom_xyz_custom: {custom_xyz_custom}')
#
#     sd = {}
#     for i in range(lambda_values.shape[0]):
#         sd[lambda_values[i]] = spectrum_approx[i]
#     sd = colour.SpectralDistribution(sd)
#     custom_xyz_colour = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)
#     logger.info(f'custom_xyz_colour: {custom_xyz_colour}')
#     logger.info(xyz / custom_xyz_colour)
#
#     # jakob
#     spectrum_jakob = colour.XYZ_to_sd(
#         xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant
#     )
#     logger.info(f'spectrum_jakob_shape: {spectrum_jakob.shape}')
#     job_xyz_custom = spectrum_to_rgb(spectrum_jakob.values)
#     logger.info(f'job_xyz_custom: {job_xyz_custom}')
#     job_xyz_colour = colour.sd_to_XYZ(spectrum_jakob, cmfs=cmfs, illuminant=illuminant)
#     logger.info(f'job_xyz_colour: {job_xyz_colour}')
#     logger.info(xyz / job_xyz_colour)
#
#     # output = '\n'
#     # for i in range(LAMBDA_COUNT + 1):
#     #     output += f'{lambda_values[i]}\t{spectrum_approx[i]}\t{spectrum_jakob[i]}\n'
#     # logger.info(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    spectral_shape = colour.SpectralShape(400, 700, 10)

    srgb = np.array([0.95, 0.8, 0.1], np.float32)
    xyz = colour.sRGB_to_XYZ(srgb)

    cmfs_cie_2 = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    cmfs = cmfs_cie_2.copy().align(spectral_shape)
    illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)

    sd = colour.XYZ_to_sd(xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
    sd_to_xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)

    logger.info(f'xyz: {xyz}')
    logger.info(f'sd_to_xyz: {sd_to_xyz / 100}')
    logger.info(f'spectrum_xyz: {spectrum_to_xyz(sd.values) / 100}')

    spectrum_initial = np.zeros((len(sd.wavelengths),), np.float32)
    spectrum_approx = coordinate_descent(
        spectrum_initial, xyz, max_iterations=100, tolerance=0.001
    )

    output = '\n'
    for i in range(len(cmfs.wavelengths)):
        output += f'{cmfs.wavelengths[i]}\t{sd[i]}\t{spectrum_approx[i]}\n'
    logger.info(output)

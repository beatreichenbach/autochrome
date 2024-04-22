import logging
import sys
from typing import Callable

import colour
import numpy as np
from PySide2 import QtWidgets

from autochrome.data.chromaticity_coordinates import COORDS
from autochrome.data.cmfs import CMFS
from autochrome.data.illuminants import ILLUMINANTS_CIE
from qt_extensions.viewer import Viewer

logger = logging.getLogger(__name__)

LAMBDA_MIN = 390
LAMBDA_MAX = 830
LAMBDA_COUNT = 21

COEFFICIENTS_COUNT = 3


def sigmoid(x: float) -> float:
    # Jakob 2019: Equation (3)
    return 0.5 + x / (2 * np.sqrt(1.0 + x**2))


def smoothstep(x: float) -> float:
    return x**2 * (3.0 - 2.0 * x)


def xy_to_xyy(xy: np.ndarray, y: float = 1) -> np.ndarray:
    xyy = np.hstack((xy, y))
    return xyy


def xyy_to_xyz(xyy: np.ndarray) -> np.ndarray:
    xyz = np.array(
        (
            xyy[0] * xyy[2] / xyy[1],
            xyy[2],
            (1 - xyy[0] - xyy[1]) * xyy[2] / xyy[1],
        )
    )
    return xyz


def xyz_to_xyy(xyz: np.ndarray) -> np.ndarray:
    xyy = np.array(
        (
            xyz[0] / np.sum(xyz),
            xyz[1] / np.sum(xyz),
            xyz[1],
        )
    )
    return xyy


def xyy_to_xy(xyy: np.ndarray) -> np.ndarray:
    return xyy[:2]


def xyz_to_xy(xyz: np.ndarray) -> np.ndarray:
    return xyy_to_xy(xyz_to_xyy(xyz))


def xyz_to_lab(xyz: np.ndarray, whitepoint: np.ndarray) -> np.array:
    x = float(xyz[0])
    y = float(xyz[1])
    z = float(xyz[2])

    xn = float(whitepoint[0])
    yn = float(whitepoint[1])
    zn = float(whitepoint[2])

    def f(t: float) -> float:
        delta = 6 / 29
        if t > delta * delta * delta:
            return np.cbrt(t)
        else:
            return t / (delta * delta * 3) + (4 / 29)

    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)

    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    lab = np.array((l, a, b))
    return lab


def eval_residual(coeffs: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    out = np.zeros(3)

    fine_samples = LAMBDA_COUNT
    # lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, fine_samples)

    for i in range(fine_samples):
        # lambda to 0..1 range
        # rel_lambda = (lambdas[i] - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
        rel_lambda = i / (fine_samples - 1)

        # polynomial
        x = 0
        for j in range(3):
            x = x * rel_lambda + coeffs[j]

        # integrate against precomputed curves
        out += xyz_table[i] * sigmoid(x)

    residual = xyz_to_lab(xyz, whitepoint) - xyz_to_lab(out, whitepoint)
    return residual


EPSILON = 0.0001


def eval_jacobian(coeffs: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # jacobian matrix
    coeffs_count = coeffs.shape[0]
    channel_count = 3
    jacobian = np.zeros((coeffs_count, coeffs_count))
    for i in range(coeffs_count):
        tmp = coeffs.copy()
        tmp[i] -= EPSILON
        r0 = eval_residual(tmp, xyz)

        tmp = coeffs.copy()
        tmp[i] += EPSILON
        r1 = eval_residual(tmp, xyz)

        for j in range(channel_count):
            jacobian[j][i] = (r1[j] - r0[j]) * (1 / (2 * EPSILON))

    return jacobian


def gauss_newton(
    xyz: np.ndarray, coefficients: np.ndarray, iterations: int = 15
) -> np.ndarray:
    threshold = 1e-6
    for i in range(iterations):
        residual = eval_residual(coefficients, xyz)
        jacobian = eval_jacobian(coefficients, xyz)

        try:
            jacobian, permutation = decompose(jacobian, 1e-15)
        except ValueError as e:
            raise ValueError('Error during decomposition.') from e

        x = solve(jacobian, permutation, residual)

        coefficients -= x
        r = np.sum(np.square(residual))

        # NOTE: It is unclear why 200 was chosen as the max. Maybe it is related to the
        #       LAB color space with a range of 256? 256 seems to give better results
        #       than 200.
        max_coefficients = np.max(coefficients)
        if max_coefficients > 200:
            coefficients *= 200 / max_coefficients

        if r < threshold:
            break

    return coefficients


def decompose(a: np.ndarray, tolerance: float) -> tuple[np.ndarray, list]:
    # lower-upper (LU) decomposition
    # a = 3x3 matrix
    n = a.shape[0]
    p = list(range(n + 1))

    for i in range(n):
        max_a = 0
        i_max = i

        for k in range(i, n):
            abs_a = np.abs(a[k][i])
            if abs_a > max_a:
                max_a = abs_a
                i_max = k

        if max_a < tolerance:
            raise ValueError

        if i_max != i:
            # pivoting p
            p[i], p[i_max] = p[i_max], p[i]
            a[i], a[i_max] = a[i_max].copy(), a[i].copy()
            p[n] += 1

        for j in range(i + 1, n):
            a[j][i] /= a[i][i]
            for k in range(i + 1, n):
                a[j][k] -= a[j][i] * a[i][k]

    return a, p


def solve(a: np.ndarray, p: list, b: np.ndarray) -> np.ndarray:
    # lower-upper (LU) solve
    n = a.shape[0]
    x = np.zeros(n)

    for i in range(n):
        x[i] = b[p[i]]

        for k in range(i):
            x[i] -= a[i][k] * x[k]

    for i in range(n - 1, -1, -1):
        for k in range(i + 1, n):
            x[i] -= a[i][k] * x[k]
        x[i] /= a[i][i]

    return x


def optimize(resolution: int) -> np.ndarray:
    scale = [smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)]
    # logger.debug(f'{scale=}')

    # channel, y, x, i?, coeffs
    data_shape = (3, resolution, resolution, resolution, 3)
    data_shape = (3 * resolution**3 * 3,)
    data = np.ndarray(data_shape)

    # l = three quadrilateral regions (Optimization 3.1)
    # shape (l, k, j, i, coeffs)
    for l in range(3):
        for j in range(resolution):
            y = j / (resolution - 1)
            for i in range(resolution):
                x = i / (resolution - 1)

                logger.debug(f'{l}, {j}, {i}')

                def iterate(start: int, end: int) -> None:
                    coefficients = np.zeros(3)
                    step = 1 if end > start else -1
                    for k in range(start, end, step):
                        b = scale[k]
                        xyz = np.zeros(3)
                        xyz[l] = b
                        xyz[(l + 1) % 3] = x * b
                        xyz[(l + 2) % 3] = y * b

                        coefficients = np.zeros(3)
                        coefficients = gauss_newton(xyz, coefficients)
                        # logger.debug(coefficients)

                        c0 = LAMBDA_MIN
                        c1 = 1 / (LAMBDA_MAX - LAMBDA_MIN)
                        a = coefficients[0]
                        b = coefficients[1]
                        c = coefficients[2]

                        index = ((l * resolution + k) * resolution + j) * resolution + i
                        # data[3 * index + 0] = a * c1**2
                        # data[3 * index + 1] = b * c1 - 2 * a * c0 * c1**2
                        # data[3 * index + 2] = c - b * c0 * c1 + a * (c0 * c1) ** 2
                        data[3 * index + 0] = a
                        data[3 * index + 1] = b
                        data[3 * index + 2] = c
                        # data[l, k, j, i, 0] = a * (c1**2)
                        # data[l, k, j, i, 1] = b * c1 - 2 * a * c0 * (c1**2)
                        # data[l, k, j, i, 2] = c - b * c0 * c1 + a * ((c0 * c1) ** 2)

                        if not l and not j and not i:
                            logger.debug(xyz)
                            logger.debug(coefficients)
                            coefficients_alt = gauss_newton(xyz, np.zeros(3))
                            logger.debug(coefficients_alt)

                # start from medium darkness and go up and down in brightness
                start = int(resolution / 5)
                iterate(start, resolution)
                iterate(start, -1)

    return data


def find_interval(values: list[float], x: float) -> int:
    # this is just an algorithm to the index that is closest to x in values.
    left = 0
    last_interval = len(values) - 2
    size = last_interval

    while size > 0:
        # half = int(size / 2)
        half = size >> 1
        middle = left + half + 1
        if values[middle] <= x:
            left = middle
            size -= half + 1
        else:
            size = half
    interval = min(last_interval, left)

    return interval


def fetch(xyz: np.ndarray, model_path: str) -> np.ndarray:
    # returns three coefficients
    # xyz must be 0..1
    coefficients_count = 3

    model = np.load(model_path)
    # resolution = model.shape[1]

    i = 0
    for j in range(1, 3):
        if xyz[j] > xyz[i]:
            i = j
    # logger.debug(f'max: {i}')

    z = float(xyz[i])
    # prevent nan values for (0, 0, 0)
    scale = (resolution - 1) / z if z > 0 else 0
    # scale = resolution - 1
    # get index
    x = float(xyz[(i + 1) % 3]) * scale
    y = float(xyz[(i + 2) % 3]) * scale

    # trilinearly interpolated lookup
    xi = min(int(x), resolution - 2)
    yi = min(int(y), resolution - 2)
    scale = [smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)]
    zi = find_interval(scale, z)

    # logger.debug('\n'.join([f'{i}: {x}' for i, x in enumerate(scale)]))
    #     logger.debug(f'x , y , z : {x}, {y}, {z}')
    #     logger.debug(f'xi, yi, zi: {xi}, {yi}, {zi}')

    #  ---

    # offset = (
    #     ((i * resolution + zi) * resolution + yi) * resolution + xi
    # ) * coefficients_count
    #
    # dx = coefficients_count
    # dy = coefficients_count * resolution
    # dz = coefficients_count * resolution**2
    #
    # x1 = x - xi
    # x0 = 1 - x1
    #
    # y1 = y - yi
    # y0 = 1 - y1
    #
    # z1 = (z - scale[zi]) / (scale[zi + 1] - scale[zi])
    # z0 = 1 - z1
    # logger.debug(f'x0, y0, z0: {x0}, {y0}, {z0}')
    # logger.debug(f'x1, y1, z1: {x1}, {y1}, {z1}')
    # logger.debug(f'{model[:3]=}')
    #
    # # turn into 1d array for lookup
    # # model = np.ravel(model)
    #
    # coefficients = np.zeros(coefficients_count)
    # for i in range(coefficients_count):
    #     tmp1 = (model[offset] * x0 + model[offset + dx] * x1) * y0
    #     tmp1 += (model[offset + dy] * x0 + model[offset + dy + dx] * x1) * y1
    #     tmp1 *= z0
    #     tmp2 = (model[offset + dz] * x0 + model[offset + dz + dx] * x1) * y0
    #     tmp2 += (model[offset + dz + dy] * x0 + model[offset + dz + dy + dx] * x1) * y1
    #     tmp2 *= z1
    #     coefficients[i] = tmp1 + tmp2
    #     offset += 1

    #     ---

    offset = (
        ((i * resolution + zi) * resolution + yi) * resolution + xi
    ) * coefficients_count
    # logger.debug(f'{i=}')

    dx = coefficients_count
    dy = coefficients_count * resolution
    dz = coefficients_count * resolution**2

    # turn into 1d array for lookup
    # model = np.ravel(model)

    xd = x - xi
    yd = y - yi
    zd = (z - scale[zi]) / (scale[zi + 1] - scale[zi])

    #     logger.debug(f'{xd=}, {yd=}, {zd=}')

    coefficients = np.zeros(coefficients_count)
    for i in range(coefficients_count):

        c000 = model[offset]
        #         logger.debug(f'{offset=}, {c000=}')
        c100 = model[offset + dx]
        c010 = model[offset + dy]
        c001 = model[offset + dz]
        c110 = model[offset + dx + dy]
        c101 = model[offset + dx + dz]
        c011 = model[offset + dy + dz]
        c111 = model[offset + dx + dy + dz]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd
        coefficients[i] = c
        offset += 1
    #     logger.debug(f'{model[:3]=}')
    #     logger.debug(f'{coefficients=}')

    return coefficients


def fma(a: float, b: float, c: float) -> float:
    return a * b + c


def eval_precise(coefficients: np.ndarray, wavelength: float) -> float:
    # get spectral value for lambda based on coefficients
    tmp = fma(coefficients[0], wavelength, coefficients[1])
    x = fma(tmp, wavelength, coefficients[2])
    y = 1 / np.sqrt(fma(x, x, 1))
    return fma(0.5 * x, y, 0.5)


# def interp(data: np.ndarray, wavelength: float):
#     # this shit literally just linearly interpolates to find the value at a wavelength
#     samples = data.shape[0]
#     # x is index of the resampled spectrum
#     x = (wavelength - LAMBDA_MIN) * ((samples - 1) / (LAMBDA_MAX - LAMBDA_MIN))
#     offset = min(int(x), samples - 2)
#     weight = x - offset
#     value = (1 - weight) * data[offset] + weight * data[offset + 1]
#     value = np.interp(wavelength, lambdas, data)
#     return
# def get_whitepoint(cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
#     samples = cmfs.shape[0]
#     fine_samples = (samples - 1) * 3 + 1
#     h = (LAMBDA_MAX - LAMBDA_MIN) / (fine_samples - 1)
#     logger.debug(h)
#
#     xyz_whitepoint = np.zeros(3)
#     for i in range(fine_samples):
#         wavelength = LAMBDA_MIN + i * h
#         xyz = interp(cmfs, wavelength)
#         illuminance = interp(illuminant, wavelength)
#         weight = (3 / 8) * h
#         if i == 0 or i == fine_samples - 1:
#             pass
#         elif (i - 1) % 3 == 2:
#             weight *= 2
#         else:
#             weight *= 3
#
#         xyz_whitepoint += xyz * illuminance * weight
#
#     # normalize ? not sure if this is correct...
#     xyz_whitepoint /= np.sum(xyz_whitepoint) / 3
#     return xyz_whitepoint
#
#
# def get_xyz_table(cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
#     samples = cmfs.shape[0]
#     fine_samples = (samples - 1) * 3 + 1
#     h = (LAMBDA_MAX - LAMBDA_MIN) / (fine_samples - 1)
#     logger.debug(h)
#
#     xyz_table = np.zeros((fine_samples, 3))
#     for i in range(fine_samples):
#         wavelength = LAMBDA_MIN + i * h
#         xyz = interp(cmfs, wavelength)
#         illuminance = interp(illuminant, wavelength)
#
#         weight = (3 / 8) * h
#         if i == 0 or i == fine_samples - 1:
#             pass
#         elif (i - 1) % 3 == 2:
#             weight *= 2
#         else:
#             weight *= 3
#
#         xyz_table[i] += xyz * illuminance * weight
#
#     # normalize ? not sure if this is correct...
#     # xyz_table /= np.sum(xyz_table, axis=0)
#
#     return xyz_table


def test_decomposition():
    a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], np.float32)
    tolerance = 1e-15
    a, permutation = decompose(a, tolerance)
    b = np.array([1, 2, 3], np.float32)
    x = solve(a, permutation, b)
    logger.debug(x)


# def colour_xyz_to_sd(xyz: np.ndarray) -> colour.SpectralDistribution:
#     # colour
#     import colour
#
#     cmfs_cie_2 = colour.MSDS_CMFS['CIE 2015 2 Degree Standard Observer']
#     d_w = 5
#     spectral_shape = colour.SpectralShape(LAMBDA_MIN, LAMBDA_MAX, d_w)
#     cmfs = cmfs_cie_2.copy().align(spectral_shape)
#
#     illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
#     sd = colour.XYZ_to_sd(xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
#     sd_to_xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100
#
#     logger.info(f'colour xyz    : {xyz}')
#     logger.info(f'colour xyz(sd): {sd_to_xyz}')
#
#     k = 1 / (np.sum(cmfs.values * illuminant.values[:, np.newaxis], axis=0))
#     sd_to_xyz = k * np.dot(sd.values * illuminant.values, cmfs.values)
#     logger.info(f'colour xyz(cu): {sd_to_xyz}')
#
#     return sd

standard_illuminant = 'A'

# chromaticity coordinates
chromaticity_coordinates = np.array(COORDS[standard_illuminant])
whitepoint = xyy_to_xyz(xy_to_xyy(chromaticity_coordinates))

# lambdas
lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT)

# cmfs
cmfs_data = CMFS['CIE 2015 2 Degree Standard Observer']
cmfs_keys = np.array(list(cmfs_data.keys()))
cmfs_values = np.array(list(cmfs_data.values()))
cmfs = np.column_stack(
    [np.interp(lambdas, cmfs_keys, cmfs_values[:, i]) for i in range(3)]
)

# illuminant
illuminant_data = ILLUMINANTS_CIE[standard_illuminant]
illuminant_keys = np.array(list(illuminant_data.keys()))
illuminant_values = np.array(list(illuminant_data.values()))
illuminant = np.interp(lambdas, illuminant_keys, illuminant_values)

xyz_table = cmfs * illuminant[:, np.newaxis]
xyz_table /= np.sum(xyz_table, axis=0)

resolution = 16

model_path = '/home/beat/dev/autochrome/autochrome/api/tasks/model_a.npy'


def main():
    # combined_spectrum = cmfs * illuminant[:, np.newaxis]
    # integral = np.trapz(combined_spectrum, axis=0)
    # whitepoint = integral / np.sum(integral)

    # logger.info(np.sum(cmfs, axis=0))
    # whitepoint = get_whitepoint(cmfs, illuminant)
    # logger.info(f'whitepoint: {whitepoint}')
    # logger.info(f'whitepoint_d65: {xyy_to_xyz(xy_to_xyy(chromaticity_coordinates))}')

    # model
    model = optimize(resolution)
    np.save(model_path, model)
    return

    # xyz
    srgb = np.array([0.9, 0.1, 0.6])
    srgb = np.array([0.2, 0.8, 0.3])
    srgb = np.array([0.6, 0.4, 1.0])
    srgb = np.array([0.0, 0.4, 0.1])
    srgb = np.array([0.1, 0.1, 0.8])
    # srgb = np.array([1.0, 1.0, 1.0])
    xyz = colour.sRGB_to_XYZ(srgb)

    logger.debug(f'whitepoint: {whitepoint}')

    coefficients = fetch(xyz, model_path)
    logger.info(f'coefficients: {coefficients}')
    # sd_rgb2spec = np.array([eval_precise(coefficients, l) for l in lambdas])

    coefficients = np.zeros(3)
    coefficients = gauss_newton(xyz, coefficients)
    logger.info(f'coefficients: {coefficients}')

    sd_rgb2spec = np.array(
        [
            eval_precise(coefficients, l / (LAMBDA_COUNT - 1))
            for l in range(LAMBDA_COUNT)
        ]
    )
    logger.debug('\n'.join([f'{lambdas[i]}: {s}' for i, s in enumerate(sd_rgb2spec)]))
    logger.info(f'rgb2spec xyz    : {xyz}')

    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)
    xyz_dot = np.dot(sd_rgb2spec * illuminant, cmfs) / k
    logger.info(f'rgb2spec xyz(dot): {xyz_dot}')

    # colour
    # sd = colour_xyz_to_sd(xyz)

    # print
    # labmdas_sd = np.linspace(LAMBDA_MIN, LAMBDA_MAX, len(sd.values))
    # sd_colour = np.interp(lambdas, labmdas_sd, sd.values)
    # output = '\n'
    # for i in range(len(lambdas)):
    #     output += f'{lambdas[i]}\t{sd_colour[i]}\t{sd_rgb2spec[i]}\n'
    # logger.info(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()

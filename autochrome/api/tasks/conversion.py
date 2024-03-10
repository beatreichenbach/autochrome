import logging
from typing import Callable

import numpy as np

from autochrome.data.chromacity_coordinates import COORDS
from autochrome.data.cmfs import CMFS
from autochrome.data.illuminants import ILLUMINANTS_CIE

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
    lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, fine_samples)

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
    # jacobian = 3x3 matrix
    jacobian = np.zeros((3, 3))
    for i in range(3):
        tmp = coeffs.copy()
        tmp[i] -= EPSILON
        r0 = eval_residual(tmp, xyz)

        tmp = coeffs.copy()
        tmp[i] += EPSILON
        r1 = eval_residual(tmp, xyz)

        for j in range(3):
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
            raise ValueError('') from e

        x = solve(jacobian, permutation, residual)

        r = 0
        for j in range(3):
            coefficients[j] -= x[j]
            r += residual[j] ** 2

        max_coefficients = np.max(coefficients)
        if max_coefficients > 200:
            coefficients *= 200 / max_coefficients

        # logger.debug(f'r: {r}')
        if r < threshold:
            break

    return coefficients


def decompose(a: np.ndarray, tolerance: float) -> tuple[np.ndarray, list]:
    # a = 3x3 matrix
    N = a.shape[0]
    p = list(range(N + 1))

    for i in range(N):
        max_a = 0
        i_max = i

        for k in range(i, N):
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
            p[N] += 1

        for j in range(i + 1, N):
            a[j][i] /= a[i][i]
            for k in range(i + 1, N):
                a[j][k] -= a[j][i] * a[i][k]

    return a, p


def solve(a: np.ndarray, p: list, b: np.ndarray) -> np.ndarray:
    N = a.shape[0]
    x = np.zeros(N)

    for i in range(N):
        x[i] = b[p[i]]

        for k in range(i):
            x[i] -= a[i][k] * x[k]

    for i in range(N - 1, -1, -1):
        for k in range(i + 1, N):
            x[i] -= a[i][k] * x[k]
        x[i] /= a[i][i]

    return x


def optimize(resolution: int) -> np.ndarray:
    scale = [smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)]

    # channel, y, x, i?, coeffs
    data_shape = (3, resolution, resolution, resolution, 3)
    data = np.ndarray(data_shape)

    # l = channels (r,g,b)/(x,y,z)
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
                        xyz = np.array((1, x, y)) * scale[k]
                        coefficients = gauss_newton(xyz, coefficients)
                        # logger.debug(coefficients)

                        c0 = LAMBDA_MIN
                        c1 = 1 / (LAMBDA_MAX - LAMBDA_MIN)
                        a = coefficients[0]
                        b = coefficients[1]
                        c = coefficients[2]

                        # index = ((l * resolution + k) * resolution + j) * resolution + i
                        data[l, k, j, i, 0] = a * (c1**2)
                        data[l, k, j, i, 1] = b * c1 - 2 * a * c0 * (c1**2)
                        data[l, k, j, i, 2] = c - b * c0 * c1 + a * ((c0 * c1) ** 2)

                start = int(resolution / 5)
                iterate(start, resolution)
                iterate(start, -1)

    return data


# def xyz_to_sd(xyz: np.ndarray, cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
#     # xyz needs to be [0,1]
#     #
#     coefficients = optimize(xyz, cmfs, illuminant)
#     sd = np.ndarray((1,))
#     return sd


# def sd_to_xyz(sd: np.ndarray, cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
#     xyz = np.ndarray([0, 0, 0])
#     return xyz


def interp(data: np.ndarray, wavelength: float):
    samples = data.shape[0]
    x = (wavelength - LAMBDA_MIN) * ((samples - 1) / (LAMBDA_MAX - LAMBDA_MIN))
    offset = min(int(x), samples - 2)
    weight = x - offset
    return (1 - weight) * data[offset] + weight * data[offset + 1]


def get_whitepoint(cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    samples = cmfs.shape[0]
    fine_samples = (samples - 1) * 3 + 1
    h = (LAMBDA_MAX - LAMBDA_MIN) / (fine_samples - 1)
    logger.debug(h)

    xyz_whitepoint = np.zeros(3)
    for i in range(fine_samples):
        wavelength = LAMBDA_MIN + i * h
        xyz = interp(cmfs, wavelength)
        illuminance = interp(illuminant, wavelength)
        # logger.debug(f'xyz: {xyz}, i: {illuminance}')

        weight = (3 / 8) * h
        if i == 0 or i == fine_samples - 1:
            pass
        elif (i - 1) % 3 == 2:
            weight *= 2
        else:
            weight *= 3

        # for k in range(3):
        #     rgb_tbl[k] += rgb *xyz * illuminance * weight
        xyz_whitepoint += xyz * illuminance * weight

    # normalize ? not sure if this is correct...
    xyz_whitepoint /= np.sum(xyz_whitepoint) / 3
    return xyz_whitepoint


# chromacity coordinates
chromacity_coordinates = np.array(COORDS['D65'])
whitepoint = xyy_to_xyz(xy_to_xyy(chromacity_coordinates))

lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT)
cmfs_data = CMFS['CIE 2015 2 Degree Standard Observer']
cmfs_keys = np.array(list(cmfs_data.keys()))
cmfs_values = np.array(list(cmfs_data.values()))
xyz_table = np.column_stack(
    [np.interp(lambdas, cmfs_keys, cmfs_values[:, i]) for i in range(3)]
)


def main():
    # lambdas = np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_COUNT)

    # cmfs
    # cmfs_data = CMFS['CIE 2015 2 Degree Standard Observer']
    # cmfs_keys = np.array(list(cmfs_data.keys()))
    # cmfs_values = np.array(list(cmfs_data.values()))
    # cmfs = np.column_stack(
    #     [np.interp(lambdas, cmfs_keys, cmfs_values[:, i]) for i in range(3)]
    # )
    # logger.info(cmfs)

    # illuminant
    # illuminant_data = ILLUMINANTS_CIE['D65']
    # illuminant_keys = np.array(list(illuminant_data.keys()))
    # illuminant_values = np.array(list(illuminant_data.values()))
    # illuminant = np.interp(lambdas, illuminant_keys, illuminant_values)
    # logger.info(illuminant)

    # combined_spectrum = cmfs * illuminant[:, np.newaxis]
    # integral = np.trapz(combined_spectrum, axis=0)
    # whitepoint = integral / np.sum(integral)

    # logger.info(np.sum(cmfs, axis=0))
    # whitepoint = get_whitepoint(cmfs, illuminant)
    # logger.info(f'whitepoint: {whitepoint}')
    # logger.info(f'whitepoint_d65: {xyy_to_xyz(xy_to_xyy(chromacity_coordinates))}')

    # xyz
    xyz = np.array([0.2, 0.4, 0.6])
    logger.info(xyz)

    # lab = xyz_to_lab(xyz, illuminant=chromacity_coordinates)
    # logger.info(lab)

    resolution = 16
    model = optimize(resolution)
    logger.debug(model)

    # test lu decomposition
    # a = np.array([[1, 2, 2], [4, 4, 2], [4, 6, 4]], np.float32)
    # tolerance = 1e-15
    # a, permutation = decompose(a, tolerance)
    # b = np.array([1, 2, 3], np.float32)
    # x = solve(a, permutation, b)
    # logger.debug(x)

    # # sd
    # sd = xyz_to_sd(xyz, cmfs, illuminant)
    #
    # # xyz
    # xyz = sd_to_xyz(sd, cmfs, illuminant)
    # logger.info(xyz)

    # colour
    # import colour
    # cmfs_cie_2 = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    # spectral_shape = colour.SpectralShape(400, 700, 20)
    # cmfs = cmfs_cie_2.copy().align(spectral_shape)
    #
    # illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    #
    # sd = colour.XYZ_to_sd(xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
    # sd_to_xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100
    #
    # logger.info(f'xyz: {xyz}')
    # logger.info(f'sd_to_xyz: {sd_to_xyz}')
    # logger.info(f'sd: {sd.values}')
    #
    # output = '\n'
    # for i in range(len(cmfs.wavelengths)):
    #     output += f'{cmfs.wavelengths[i]}\t{sd.values[i]}\n'
    # logger.info(output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()

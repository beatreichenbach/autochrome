import hashlib
import logging
import os
from functools import lru_cache

import numpy as np

from autochrome.api.data import Project
from autochrome.data.chromaticity_coordinates import COORDS
from autochrome.data.cmfs import CMFS
from autochrome.data.illuminants import ILLUMINANTS_CIE
from autochrome.utils import color

logger = logging.getLogger(__name__)

EPSILON = 0.0001
COEFFICIENTS_COUNT = 3
LAMBDA_MIN = 390
LAMBDA_MAX = 830


def sigmoid(x: float) -> float:
    # Jakob 2019: Equation (3)
    return 0.5 + x / (2 * np.sqrt(1.0 + x**2))


def smoothstep(x: float) -> float:
    return x**2 * (3.0 - 2.0 * x)


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


def eval_residual(
    coeffs: np.ndarray, xyz: np.ndarray, xyz_table: np.ndarray, whitepoint: np.ndarray
) -> np.ndarray:
    coeffs_count = coeffs.shape[0]
    out = np.zeros(xyz.shape)
    lambda_count = xyz_table.shape[0]

    for i in range(lambda_count):
        # lambda to 0..1 range
        rel_lambda = i / (lambda_count - 1)

        # polynomial
        x = 0
        for j in range(coeffs_count):
            x = x * rel_lambda + coeffs[j]

        # integrate against precomputed curves
        out += xyz_table[i] * sigmoid(x)

    residual = color.xyz_to_lab(xyz, whitepoint) - color.xyz_to_lab(out, whitepoint)
    return residual


def eval_jacobian(
    coeffs: np.ndarray, xyz: np.ndarray, xyz_table: np.ndarray, whitepoint: np.ndarray
) -> np.ndarray:
    # jacobian matrix
    coeffs_count = coeffs.shape[0]
    channel_count = xyz.shape[0]
    jacobian = np.zeros((coeffs_count, coeffs_count))
    for i in range(coeffs_count):
        tmp = coeffs.copy()
        tmp[i] -= EPSILON
        r0 = eval_residual(tmp, xyz, xyz_table, whitepoint)

        tmp = coeffs.copy()
        tmp[i] += EPSILON
        r1 = eval_residual(tmp, xyz, xyz_table, whitepoint)

        for j in range(channel_count):
            jacobian[j][i] = (r1[j] - r0[j]) * (1 / (2 * EPSILON))

    return jacobian


def gauss_newton(
    xyz: np.ndarray,
    coefficients: np.ndarray,
    xyz_table: np.ndarray,
    whitepoint: np.ndarray,
    iterations: int = 15,
) -> np.ndarray:
    threshold = 1e-6
    for i in range(iterations):
        residual = eval_residual(coefficients, xyz, xyz_table, whitepoint)
        jacobian = eval_jacobian(coefficients, xyz, xyz_table, whitepoint)

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


def get_scale(resolution: int) -> tuple[float, ...]:
    return tuple(
        smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)
    )


def create_model(
    resolution: int,
    xyz_table: np.ndarray,
    whitepoint: np.ndarray,
) -> np.ndarray:
    scale = get_scale(resolution)
    # logger.debug(f'{scale=}')

    # channel, y, x, i?, coeffs
    data_shape = (3, resolution, resolution, resolution, 3)
    data_shape = (3 * resolution**3 * 3,)
    data = np.ndarray(data_shape)

    # l = three quadrilateral regions (Optimization 3.1)
    # shape (l, k, j, i, coeffs)

    iteration_max = (3 * resolution * resolution) - 1

    for l in range(3):
        for j in range(resolution):
            y = j / (resolution - 1)
            for i in range(resolution):
                x = i / (resolution - 1)

                iteration = ((l * resolution + j) * resolution) + i
                progress = iteration / iteration_max
                logger.info(f'Creating model: {100 * progress:>6.2f}%')

                def iterate(start: int, end: int) -> None:
                    # coefficients = np.zeros(3)
                    step = 1 if end > start else -1
                    for k in range(start, end, step):
                        b = scale[k]
                        xyz = np.zeros(3)
                        xyz[l] = b
                        xyz[(l + 1) % 3] = x * b
                        xyz[(l + 2) % 3] = y * b

                        # NOTE: initializing coefficients at 0 produced better results
                        #       than starting from previous result. Why?
                        coefficients = np.zeros(COEFFICIENTS_COUNT)
                        coefficients = gauss_newton(
                            xyz, coefficients, xyz_table, whitepoint
                        )
                        # logger.debug(coefficients)

                        # c0 = LAMBDA_MIN
                        # c1 = 1 / (LAMBDA_MAX - LAMBDA_MIN)
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

                        # if not l and not j and not i:
                        #     logger.debug(xyz)
                        #     logger.debug(coefficients)
                        #     coefficients_alt = gauss_newton(
                        #         xyz, np.zeros(COEFFICIENTS_COUNT), xyz_table, whitepoint
                        #     )
                        #     logger.debug(coefficients_alt)

                # start from medium darkness and go up and down in brightness
                start = int(resolution / 5)
                iterate(start, resolution)
                iterate(start, -1)

    return data


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


def fetch(model: np.ndarray, resolution: int, xyz: np.ndarray) -> np.ndarray:
    # returns three coefficients
    # xyz must be 0..1

    # resolution = model.shape[1]

    i = 0
    for j in range(1, 3):
        if xyz[j] > xyz[i]:
            i = j

    z = float(xyz[i])
    # prevent nan values for (0, 0, 0)
    normalize = (resolution - 1) / z if z > 0 else 0
    # get index
    x = float(xyz[(i + 1) % 3]) * normalize
    y = float(xyz[(i + 2) % 3]) * normalize

    xi = min(int(x), resolution - 2)
    yi = min(int(y), resolution - 2)
    scale = [smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)]
    zi = find_interval(scale, z)

    offset = (
        ((i * resolution + zi) * resolution + yi) * resolution + xi
    ) * COEFFICIENTS_COUNT

    dx = COEFFICIENTS_COUNT
    dy = COEFFICIENTS_COUNT * resolution
    dz = COEFFICIENTS_COUNT * resolution**2

    xd = x - xi
    yd = y - yi
    zd = (z - scale[zi]) / (scale[zi + 1] - scale[zi])

    # NOTE: Trilateral interpolation
    #       https://en.wikipedia.org/wiki/Trilinear_interpolation
    coefficients = np.zeros(COEFFICIENTS_COUNT)
    for i in range(COEFFICIENTS_COUNT):
        c000 = model[offset]
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

    return coefficients


def find_interval(values: list[float], x: float) -> int:
    # Return the index that is closest to x in values.
    left = 0
    last_interval = len(values) - 2
    size = last_interval

    while size > 0:
        half = size >> 1
        middle = left + half + 1
        if values[middle] <= x:
            left = middle
            size -= half + 1
        else:
            size = half
    interval = min(last_interval, left)

    return interval


def fma(a: float, b: float, c: float) -> float:
    return a * b + c


def eval_precise(coefficients: np.ndarray, wavelength: float) -> float:
    # get spectral value for lambda based on coefficients
    c = fma(coefficients[0], wavelength, coefficients[1])
    x = fma(c, wavelength, coefficients[2])
    y = 1 / np.sqrt(fma(x, x, 1))
    return fma(0.5 * x, y, 0.5)


def get_cmfs(variation: str, lambdas: np.ndarray) -> np.ndarray:
    cmfs_data = CMFS[variation]
    cmfs_keys = np.array(list(cmfs_data.keys()))
    cmfs_values = np.array(list(cmfs_data.values()))
    cmfs = np.column_stack(
        [np.interp(lambdas, cmfs_keys, cmfs_values[:, i]) for i in range(3)]
    )
    return cmfs


def get_illuminant(standard_illuminant: str, lambdas: np.ndarray) -> np.ndarray:
    illuminant_data = ILLUMINANTS_CIE[standard_illuminant]
    illuminant_keys = np.array(list(illuminant_data.keys()))
    illuminant_values = np.array(list(illuminant_data.values()))
    illuminant = np.interp(lambdas, illuminant_keys, illuminant_values)
    return illuminant


@lru_cache(1)
def get_whitepoint(standard_illuminant: str) -> np.ndarray:
    chromaticity_coordinates = np.array(COORDS[standard_illuminant])
    whitepoint = color.xyy_to_xyz(color.xy_to_xyy(chromaticity_coordinates))
    return whitepoint


class SpectralTask:

    @lru_cache(1)
    def update_xyz_table(
        self,
        cmfs_variation: str,
        standard_illuminant: str,
        lambda_count: int,
        lambda_min: int,
        lambda_max: int,
    ) -> np.ndarray:
        lambdas = np.linspace(lambda_min, lambda_max, lambda_count)
        cmfs = get_cmfs(cmfs_variation, lambdas)
        illuminant = get_illuminant(standard_illuminant, lambdas)
        xyz_table = cmfs * illuminant[:, np.newaxis]
        xyz_table /= np.sum(xyz_table, axis=0)
        return xyz_table

    @lru_cache(10)
    def update_model_name(self, *args):
        h = hashlib.md5()
        for arg in args:
            h.update(str.encode(str(arg)))
        return h.hexdigest()

    def cache_model(self, resolution: int, lambda_count: int) -> None:
        standard_illuminant = 'D65'
        cmfs_variation = 'CIE 2015 2 Degree Standard Observer'
        lambda_min = LAMBDA_MIN
        lambda_max = LAMBDA_MAX

        model_args = (
            resolution,
            lambda_count,
            lambda_min,
            lambda_max,
            standard_illuminant,
            cmfs_variation,
        )
        logger.debug(f'{model_args=}')

        model_name = self.update_model_name(model_args)
        logger.debug(f'{model_name=}')

        filename = f'{model_name}.npy'
        model_path = os.path.join(
            '/home/beat/dev/autochrome/autochrome/api/tasks', filename
        )
        logger.debug(f'{model_path=}')

        whitepoint = get_whitepoint(standard_illuminant=standard_illuminant)
        xyz_table = self.update_xyz_table(
            cmfs_variation=cmfs_variation,
            standard_illuminant=standard_illuminant,
            lambda_count=lambda_count,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

        logger.info(f'Caching model...')
        model = create_model(resolution, xyz_table, whitepoint)
        np.save(model_path, model)
        logger.info(f'Model saved: {model_path}')

    def run(self, project: Project) -> None:
        self.cache_model(
            resolution=project.emulsion.model_resolution,
            lambda_count=project.emulsion.lambda_count,
        )

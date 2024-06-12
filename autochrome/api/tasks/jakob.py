import hashlib
import logging
import os
from functools import lru_cache

import numpy as np

from autochrome.api.data import Project
from autochrome.storage import Storage
from autochrome.utils import color
from autochrome.utils.timing import timer

EPSILON = 0.0001
COEFFICIENTS_COUNT = 3
LAMBDA_MIN = 390
LAMBDA_MAX = 830

logger = logging.getLogger(__name__)
storage = Storage()


def hash_args(*args) -> str:
    h = hashlib.md5()
    for arg in args:
        h.update(str.encode(str(arg)))
    return h.hexdigest()


@lru_cache(10)
def get_model_path(project: Project, cache_dir: str = '') -> str:
    args = (
        project.emulsion.model_resolution,
        project.emulsion.wavelength_count,
        project.emulsion.standard_illuminant,
        project.emulsion.cmfs_variation,
        LAMBDA_MIN,
        LAMBDA_MAX,
    )
    model_name = hash_args(*args)
    if not cache_dir:
        cache_dir = os.path.join(storage.cache_dir, 'models')

    filename = f'{model_name}.npy'
    model_path = os.path.join(cache_dir, filename)
    logger.debug(f'{model_path=}')
    return model_path


def sigmoid(x: float) -> float:
    # Jakob 2019: Equation (3)
    return 0.5 + x / (2 * np.sqrt(1.0 + x**2))


def smoothstep(x: float) -> float:
    return x**2 * (3.0 - 2.0 * x)


def decompose(a: np.ndarray, tolerance: float) -> tuple[np.ndarray, list]:
    """Return a lower-upper (LU) decomposition.
    `a` is a 3x3 matrix
    """
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
    """Solve lower-upper (LU) decomposition."""
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
    """Return the Jacobian Matrix."""
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
    """Solve non-linear least squares problem with Gauss-Newton algorithm."""
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
    """Return a smooth interpolation between 0 and 1."""
    return tuple(
        smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)
    )


def create_model(
    resolution: int,
    xyz_table: np.ndarray,
    whitepoint: np.ndarray,
) -> np.ndarray:
    """Return a model of coefficients for three quadrilateral regions based on
    Jakob 2019."""
    scale = get_scale(resolution)

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
    """Trilinear Interpolation given a model with 3 dimensions."""
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
    """Returns three coefficients from a model.
    xyz must be 0..1.
    """

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
    """Return the index that is closest to x in values."""
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
    """Get spectral value for lambda based on coefficients."""
    c = fma(coefficients[0], wavelength, coefficients[1])
    x = fma(c, wavelength, coefficients[2])
    y = 1 / np.sqrt(fma(x, x, 1))
    return fma(0.5 * x, y, 0.5)


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
        cmfs = color.get_cmfs(cmfs_variation, lambdas)
        illuminant = color.get_illuminant(standard_illuminant, lambdas)
        xyz_table = cmfs * illuminant[:, np.newaxis]
        xyz_table /= np.sum(xyz_table, axis=0)
        return xyz_table

    def cache_model(
        self,
        resolution: int,
        model_path: str,
        cmfs_variation: str,
        standard_illuminant: str,
        lambda_count: int,
        lambda_min: int,
        lambda_max: int,
    ) -> None:

        whitepoint = color.get_whitepoint(standard_illuminant=standard_illuminant)
        xyz_table = self.update_xyz_table(
            cmfs_variation=cmfs_variation,
            standard_illuminant=standard_illuminant,
            lambda_count=lambda_count,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

        logger.info(f'Caching model...')
        model = create_model(resolution, xyz_table, whitepoint)

        cache_dir = os.path.dirname(model_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        np.save(model_path, model)
        logger.info(f'Model saved: {model_path}')

    @timer
    def run(self, project: Project) -> None:
        model_path = get_model_path(project)
        if os.path.exists(model_path):
            logger.debug('Model already exists.')
            return

        lambda_min = LAMBDA_MIN
        lambda_max = LAMBDA_MAX
        self.cache_model(
            resolution=project.emulsion.model_resolution,
            model_path=model_path,
            cmfs_variation=project.emulsion.cmfs_variation,
            standard_illuminant=project.emulsion.standard_illuminant,
            lambda_count=project.emulsion.wavelength_count,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

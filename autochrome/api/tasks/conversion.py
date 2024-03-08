import logging

import colour
import numpy as np

from autochrome.data.chromacity_coordinates import COORDS
from autochrome.data.cmfs import CMFS
from autochrome.data.illuminants import ILLUMINANTS_CIE

logger = logging.getLogger(__name__)

LAMBDA_MIN = 400
LAMBDA_MAX = 700
LAMBDA_COUNT = 15


# def whitepoint(cmfs: np.array, illuminant: np.array) -> np.array:
#     for


def sigmoid(x: float) -> float:
    # Jakob 2019: Equation (3)
    return 0.5 + x / (2 * np.sqrt(1.0 + x * x))


def xy_to_xyy(xy: np.ndarray, y: float = 1) -> np.ndarray:
    xyy = np.hstack((xy, y))
    return xyy


def xyy_to_xyz(xyy: np.ndarray) -> np.ndarray:
    xyz = np.array(
        (xyy[0] * xyy[2] / xyy[1], xyy[2], (1 - xyy[0] - xyy[1]) * xyy[2] / xyy[1])
    )
    return xyz


def xyz_to_lab(xyz: np.ndarray, illuminant: np.ndarray) -> np.array:

    x = float(xyz[0])
    y = float(xyz[1])
    z = float(xyz[2])
    whitepoint = xyy_to_xyz(xy_to_xyy(illuminant))
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


#
# def main():
#     res = 10
#     for l in range(3):
#         for j in range(res):


def solve_coefficients(
    xyz: np.ndarray, cmfs: np.ndarray, illuminant: np.ndarray
) -> np.ndarray:
    coefficients = np.ndarray([0, 0, 0])
    return coefficients


def xyz_to_sd(xyz: np.ndarray, cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    # xyz needs to be [0,1]
    #
    coefficients = solve_coefficients(xyz, cmfs, illuminant)
    sd = np.ndarray((1,))
    return sd


def sd_to_xyz(sd: np.ndarray, cmfs: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    xyz = np.ndarray([0, 0, 0])
    return xyz


def main():
    cmfs_data = CMFS['CIE 2015 2 Degree Standard Observer']

    wavelengths = np.linspace(400, 700, 301)

    # cmfs
    cmfs_keys = np.array(list(cmfs_data.keys()))
    cmfs_values = np.array(list(cmfs_data.values()))
    cmfs = np.column_stack(
        [np.interp(wavelengths, cmfs_keys, cmfs_values[:, i]) for i in range(3)]
    )
    # logger.info(cmfs)

    # illuminant
    illuminant_data = ILLUMINANTS_CIE['D65']
    illuminant_keys = np.array(list(illuminant_data.keys()))
    illuminant_values = np.array(list(illuminant_data.values()))
    illuminant = np.interp(wavelengths, illuminant_keys, illuminant_values)
    #     logger.info(illuminant)

    combined_spectrum = cmfs * illuminant[:, np.newaxis]
    integral = np.trapz(combined_spectrum, axis=0)
    whitepoint = integral / np.sum(integral)
    # whitepoint = np.mean(combined_spectrum / 100, axis=0)

    # chromacity coordinates
    chromacity_coordinates = np.array(COORDS['D65'])

    logger.info(whitepoint)

    # xyz
    xyz = np.array([0.2, 0.4, 0.6])
    logger.info(xyz)

    lab = xyz_to_lab(xyz, illuminant=chromacity_coordinates)
    logger.info(lab)

    # # sd
    # sd = xyz_to_sd(xyz, cmfs, illuminant)
    #
    # # xyz
    # xyz = sd_to_xyz(sd, cmfs, illuminant)
    # logger.info(xyz)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()

    # if __name__ == '__main__':
    #     logging.basicConfig(level=logging.DEBUG)
    #
    #     spectral_shape = colour.SpectralShape(400, 700, 20)
    #
    #     srgb = np.array([0, 1.0, 0], np.float32)
    #     xyz = colour.sRGB_to_XYZ(srgb)
    #
    #     cmfs_cie_2 = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    #     cmfs = cmfs_cie_2.copy().align(spectral_shape)
    # import colour
    #
    # illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
#
#     sd = colour.XYZ_to_sd(xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
#     sd_to_xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100
#
#     logger.info(f'xyz: {xyz}')
#     logger.info(f'sd_to_xyz: {sd_to_xyz}')
#     # logger.info(f'spectrum_xyz: {spectrum_to_xyz(sd.values)}')
#
#     logger.info(f'sd: {sd.values}')
#
#     # LUT = colour.recovery.LUT3D_Jakob2019()
#     # LUT.generate(colour.models.RGB_COLOURSPACE_sRGB, cmfs, illuminant, 4, lambda x: x)
#     # LUT.write('lut.coeff')
#     # sd_lut = LUT.RGB_to_sd(srgb, cmfs.shape)
#
#     output = '\n'
#     for i in range(len(cmfs.wavelengths)):
#         output += f'{cmfs.wavelengths[i]}\t{sd.values[i]}\n'
#     logger.info(output)

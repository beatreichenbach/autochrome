import logging
import sys

import numpy as np
from PySide2 import QtWidgets
from qt_extensions import theme
from qt_extensions.viewer import Viewer

from autochrome.api.data import Project
from autochrome.api.tasks import opencl, jakob, emulsion
from autochrome.utils import color

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


def reconstruct_model(array: np.ndarray) -> np.ndarray:
    project = Project()
    wavelength_count = project.emulsion.wavelength_count

    model_path = jakob.get_model_path(project)
    model = np.load(model_path)
    lambdas = np.linspace(
        project.emulsion.lambda_min, project.emulsion.lambda_max, wavelength_count
    )
    cmfs = color.get_cmfs(project.emulsion.cmfs_variation, lambdas)
    illuminant = color.get_illuminant(project.emulsion.standard_illuminant, lambdas)

    reconstructed = np.zeros(array.shape)
    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            rgb = array[y, x]
            xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)

            coefficients = jakob.fetch(model, project.emulsion.model_resolution, xyz)
            sd = np.array(
                [
                    jakob.eval_precise(coefficients, w / (wavelength_count - 1))
                    for w in range(wavelength_count)
                ]
            )

            xyz = np.dot(sd * illuminant, cmfs) / k
            rgb = np.clip(np.dot(XYZ_TO_SRGB, xyz), 0, 1)
            reconstructed[y, x] = xyz
    return reconstructed


def reconstruct(array: np.ndarray) -> np.ndarray:
    project = Project()
    wavelength_count = project.emulsion.wavelength_count
    standard_illuminant = project.emulsion.standard_illuminant

    spectral_task = jakob.SpectralTask()
    whitepoint = color.get_whitepoint(standard_illuminant=standard_illuminant)
    xyz_table = spectral_task.update_xyz_table(
        cmfs_variation=project.emulsion.cmfs_variation,
        standard_illuminant=standard_illuminant,
        lambda_count=wavelength_count,
        lambda_min=project.emulsion.lambda_min,
        lambda_max=project.emulsion.lambda_max,
    )

    lambdas = np.linspace(
        project.emulsion.lambda_min, project.emulsion.lambda_max, wavelength_count
    )
    cmfs = color.get_cmfs(project.emulsion.cmfs_variation, lambdas)
    illuminant = color.get_illuminant(project.emulsion.standard_illuminant, lambdas)

    reconstructed = np.zeros(array.shape)
    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            rgb = array[y, x]
            xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)

            coefficients = np.zeros(jakob.COEFFICIENTS_COUNT)
            coefficients = jakob.gauss_newton(xyz, coefficients, xyz_table, whitepoint)
            sd = np.array(
                [
                    jakob.eval_precise(coefficients, w / (wavelength_count - 1))
                    for w in range(wavelength_count)
                ]
            )

            xyz = np.dot(sd * illuminant, cmfs) / k
            rgb = np.clip(np.dot(XYZ_TO_SRGB, xyz), 0, 1)
            reconstructed[y, x] = xyz
    return reconstructed


def reconstruct_scipy(array: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    class EarlyExitException(Exception):
        def __init__(self, coefficients, error):
            super().__init__()
            self.coefficients = coefficients
            self.error = error

    def error_functions(
        coeffs: np.ndarray,
        xyz: np.ndarray,
        xyz_table: np.ndarray,
        whitepoint: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        residual = jakob.eval_residual(coeffs, xyz, xyz_table, whitepoint)
        max_error = 0.023

        error = np.sqrt(np.sum(residual**2))
        if max_error is not None and error <= max_error:
            raise EarlyExitException(coeffs, error)

        jacobian = jakob.eval_jacobian(coeffs, xyz, xyz_table, whitepoint)
        d_error = jakob.lu_solve(jacobian, residual)

        return error, d_error

    project = Project()
    wavelength_count = project.emulsion.wavelength_count
    standard_illuminant = project.emulsion.standard_illuminant

    spectral_task = jakob.SpectralTask()
    whitepoint = color.get_whitepoint(standard_illuminant=standard_illuminant)
    xyz_table = spectral_task.update_xyz_table(
        cmfs_variation=project.emulsion.cmfs_variation,
        standard_illuminant=standard_illuminant,
        lambda_count=wavelength_count,
        lambda_min=project.emulsion.lambda_min,
        lambda_max=project.emulsion.lambda_max,
    )

    lambdas = np.linspace(
        project.emulsion.lambda_min, project.emulsion.lambda_max, wavelength_count
    )
    cmfs = color.get_cmfs(project.emulsion.cmfs_variation, lambdas)
    illuminant = color.get_illuminant(project.emulsion.standard_illuminant, lambdas)

    reconstructed = np.zeros(array.shape)
    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            rgb = array[y, x]
            xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)

            coefficients = np.zeros(jakob.COEFFICIENTS_COUNT)

            try:
                result = minimize(
                    error_functions,
                    coefficients,
                    (xyz, xyz_table, whitepoint),
                    method="L-BFGS-B",
                    jac=True,
                )
                coefficients = result.x
            except EarlyExitException as e:
                coefficients = e.coefficients

            sd = np.array(
                [
                    jakob.eval_precise(coefficients, w / (wavelength_count - 1))
                    for w in range(wavelength_count)
                ]
            )

            xyz = np.dot(sd * illuminant, cmfs) / k
            rgb = np.clip(np.dot(XYZ_TO_SRGB, xyz), 0, 1)
            reconstructed[y, x] = rgb
    return reconstructed


def reconstruct_colour(array: np.ndarray) -> np.ndarray:
    from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape, XYZ_to_sd
    from colour.colorimetry import sd_to_XYZ_integration

    cmfs = (
        MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy()
        .align(SpectralShape(360, 780, 10))
    )
    illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)

    reconstructed = np.zeros(array.shape)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            rgb = array[y, x]
            xyz = np.clip(np.dot(SRGB_TO_XYZ, rgb), 0, 1)

            sd = XYZ_to_sd(xyz, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
            xyz = sd_to_XYZ_integration(sd, cmfs, illuminant) / 100

            rgb = np.clip(np.dot(XYZ_TO_SRGB, xyz), 0, 1)
            reconstructed[y, x] = rgb
    return reconstructed


def test_pattern_diff():
    project = Project()
    spectral_task = jakob.SpectralTask()
    spectral_task.run(project=project)

    pattern = test_pattern(5)
    pattern_reconstructed = reconstruct_model(pattern)
    # pattern_reconstructed = reconstruct(pattern)
    # pattern_reconstructed = reconstruct_colour(pattern)
    # pattern_reconstructed = reconstruct_scipy(pattern)

    xyz = np.zeros(pattern.shape)
    for y in range(pattern.shape[0]):
        for x in range(pattern.shape[1]):
            xyz[y, x] = np.clip(np.dot(SRGB_TO_XYZ, pattern[y, x]), 0, 1)
    pattern = xyz

    diff = np.abs(pattern - pattern_reconstructed)
    logging.info(f'Difference mean: {np.mean(diff)}')
    output = np.vstack((pattern, pattern_reconstructed, diff))

    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(theme.modern_dark)

    viewer = Viewer()
    viewer.show()
    viewer.set_array(output)
    viewer.move(
        viewer.frameGeometry().topLeft()
        + (viewer.screen().geometry().center() - viewer.frameGeometry().center())
    )

    sys.exit(app.exec_())


def test_emulsion_task():
    queue = opencl.command_queue()

    project = Project()
    project.input.image_path = r'C:\Users\Beat\Downloads\Jorge_Martin_at_the_2023_Japanese_motorcycle_Grand_Prix.jpg'
    project.input.colorspace = 'Output - sRGB'
    project.render.force_resolution = False

    spectral_task = jakob.SpectralTask()
    spectral_task.run(project=project)

    emulsion_task = emulsion.EmulsionTask(queue)
    spectral_images = emulsion_task.run(project=project)

    output = (
        spectral_images[0].array + spectral_images[1].array + spectral_images[2].array
    )

    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(theme.modern_dark)

    viewer = Viewer()
    viewer.set_array(output)
    viewer.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_pattern_diff()

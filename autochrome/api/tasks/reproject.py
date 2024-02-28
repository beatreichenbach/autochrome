import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

ANALOG_CHANNEL_GAIN = np.array([2.2933984, 1, 1.62308182])
TYPICAL_SCENE_REFLECTIVITY = 0.18
TYPICAL_SCENE_REFLECTIVITY = 1


def load_ms_filter(csv):
    """
    Load a 16 channel Multi-Spectral filter in the format specified by
    'resources/MS_Camera_QE.csv'

    :param csv: path to source CSV file
    :return: filter, filter bands, and filter peaks as numpy arrays
    """
    df = pd.read_csv(csv, skiprows=[1])
    ms_filter = df.iloc[:, 1:17].to_numpy()
    bands = df['Channel'].to_numpy()

    df = pd.read_csv(csv)
    ms_peaks = df.iloc[0, 1:17].to_numpy().astype(np.int16)

    return ms_filter, bands, ms_peaks


def load_rgb_filter(csv):
    """
    Load a three channel RGB filter in the format specified by
    'resources/RGB_Camera_QE.csv'

    :param csv: path to source CSV file
    :return: camera filter and filter bands as numpy arrays
    """
    df = pd.read_csv(csv)
    camera_filter = df[['R', 'G1', 'B']].to_numpy() * ANALOG_CHANNEL_GAIN
    bands = df['Wavelength[nm]'].to_numpy()
    return camera_filter, bands


def load_rgb_filter2(csv, bands):
    """
    Load a three channel RGB filter in the format specified by
    'resources/RGB_Camera_QE.csv'

    :param csv: path to source CSV file
    :return: camera filter and filter bands as numpy arrays
    """
    df = pd.read_csv(csv)
    df = df[df['Wavelength[nm]'].isin(bands)]
    camera_filter = df[['R', 'G1', 'B']].to_numpy() * ANALOG_CHANNEL_GAIN
    return camera_filter


def make_spectral_bands(nm_start, nm_stop, nm_step, dtype=np.int32):
    """
    Boilerplate code to make a uniform spectral wavelength range

    :param nm_start: start wavelength in [nm]
    :param nm_stop:  stop wavelength (inclusive) in [nm]
    :param nm_step: spectral resolution in [nm]
    :param dtype: default - integer

    :return: numpy array of wavelengths
    """
    if nm_step <= 0:
        raise ValueError("make_spectral_bands: step must be positive.")
    return np.arange(
        start=nm_start,
        stop=nm_stop + nm_step / 2,  # make sure to include the stop wavelength
        step=nm_step,
    ).astype(dtype)


def project_cube(pixels, filters, clip_negative=False):
    """
    Project multispectral pixels to low dimension using a filter function
    (such as a camera response)

    :param pixels: numpy array of multispectral pixels, shape [..., num_hs_bands]
    :param filters: filter response, [num_hs_bands, num_mc_chans]
    :param clip_negative: whether to clip negative values

    :return: a numpy array of the projected pixels, shape [..., num_mc_chans]

    :raise: RuntimeError if `pixels` or `filters` are passed transposed

    """

    # assume the number of spectral channels match (will crash inside if not)
    if np.shape(pixels)[-1] != np.shape(filters)[0]:
        raise RuntimeError(
            f'{__file__}: projectCube - incompatible dimensions! got {np.shape(pixels)} and {np.shape(filters)}'
        )

    projected = np.matmul(pixels, filters)

    if clip_negative:
        projected = projected.clip(0, None)

    return projected


def resample_hs_picked(
    cube, bands, new_bands, interp_mode='linear', fill_value='extrapolate'
):
    """
    Resample a hyperspectral cube at picked arbitrary 'newBands'

    :param cube: numpy array of HS data, shape [H, W, num_hs_channels] or [num_samples, num of channels]
    :param bands: numpy array of the wavelength (nm) of each channel in the cube, shape [num of channels]
    :param new_bands: numpy array of the wavelength (nm) at which to resample the cube, shape [num of new bands]
    :param interp_mode: See more details in CubeUtils.resampleHS
    :param fill_value: if 'extrapolate', then data will be extrapolated,
                       if float, then the value will be set at both ends of the range
                       if tuple of floats (a, b), then a will fill the bottom of the range and b the top
                       default is NaN

    :return: a numpy array of the sampled cube, shape [H, W, num of new bands]
    """
    interp_modes = [
        'zero',
        'slinear',
        'quadratic',
        'cubic',
        'linear',
        'nearest',
        'previous',
        'next',
    ]  # taken from interp1d
    if interp_mode not in interp_modes:
        raise ValueError(
            f"resampleHSPicked: {interp_mode} is not a valid interpMode."
            f"Options are {','.join(interp_modes)}."
        )

    interp_fun = interp1d(
        bands,
        cube,
        axis=-1,
        kind=interp_mode,
        assume_sorted=True,
        fill_value=fill_value,
        bounds_error=False,
    )
    resampled = interp_fun(new_bands)

    return resampled


def project_hs(cube, cube_bands, qes, qe_bands, clip_negative, interp_mode='linear'):
    """
    Project a spectral array

    :param cube: Input hyperspectral cube
    :param cube_bands: bands of hyperspectral cube
    :param qes: filter response to use for projection
    :param qe_bands: bands of filter response
    :param clip_negative: clip values below 0
    :param interp_mode: interpolation mode for missing values
    :return:
    :return: numpy array of projected data, shape [..., num_channels ]
    """

    if not np.all(qe_bands == cube_bands):
        # then sample the qes on the data bands
        dx_qes = qe_bands[1] - qe_bands[0]
        dx_hs = cube_bands[1] - cube_bands[0]
        if np.any(np.diff(qe_bands) != dx_qes) or np.any(np.diff(cube_bands) != dx_hs):
            raise ValueError(
                f'V81Filter.projectHS - '
                f'can only interpolate from uniformly sampled bands\n'
                f'got hs bands: {cube_bands}\n'
                f'filter bands: {qe_bands}'
            )

        if dx_qes < 0:
            # we assume the qe_bands are sorted ascending inside resampleHSPicked,
            # reverse them
            qes = qes[::-1]
            qe_bands = qe_bands[::-1]

        # find the limits of the interpolation, WE DON'T WANT TO EXTRAPOLATE!
        # the limits must be defined by the data bands so the interpolated qe matches
        min_band = cube_bands[
            np.argwhere(cube_bands >= qe_bands.min()).min()
        ]  # the first data band which has a respective qe value
        max_band = cube_bands[
            np.argwhere(cube_bands <= qe_bands.max()).max()
        ]  # the last data band which has a respective qe value
        # TODO is there a minimal overlap we want to enforce?

        cube = cube[..., np.logical_and(cube_bands >= min_band, cube_bands <= max_band)]
        # shared domain with the spectral resolution of the spectral data
        shared_bands = make_spectral_bands(min_band, max_band, dx_hs)
        qes = resample_hs_picked(
            qes.T,
            bands=qe_bands,
            new_bands=shared_bands,
            interp_mode=interp_mode,
            fill_value=np.nan,
        ).T

    return project_cube(cube, qes, clip_negative=clip_negative)

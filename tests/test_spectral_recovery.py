import numpy as np
import colour
from colour.utilities import numpy_print_options


def main() -> None:
    XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    cmfs = (
        colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy()
        .align(colour.SpectralShape(360, 780, 10))
    )
    illuminant = colour.SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    sd = colour.XYZ_to_sd(XYZ, method='Jakob 2019', cmfs=cmfs, illuminant=illuminant)
    with numpy_print_options(suppress=True):
        print(sd)


if __name__ == '__main__':
    main()

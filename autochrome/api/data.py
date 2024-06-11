import enum
from dataclasses import field

from PySide2 import QtCore

from qt_extensions.typeutils import hashable_dataclass, deep_field

from autochrome.api.tasks.opencl import Image


class EngineError(ValueError):
    pass


@enum.unique
class RenderElement(enum.Enum):
    INPUT = enum.auto()
    EMULSION_R = enum.auto()
    EMULSION_G = enum.auto()
    EMULSION_B = enum.auto()
    KERNEL = enum.auto()
    HALATION = enum.auto()
    GRAIN = enum.auto()


@hashable_dataclass
class RenderImage:
    image: Image
    element: RenderElement


@hashable_dataclass
class Input:
    image_path: str = '/home/beat/Downloads/ferrari_hamilton_live_v02.exr'
    colorspace: str = 'ACES - ACEScg'


@hashable_dataclass
class Emulsion:
    wavelength_count: int = 21
    model_resolution: int = 16
    lambda_count: int = 21
    curves_file: str = '$CURVES/kodak_ektachrome_100.json'
    standard_illuminant: str = 'D65'
    cmfs_variation: str = 'CIE 2015 2 Degree Standard Observer'


@hashable_dataclass
class Halation:
    roughness: float = 0.2
    height: float = 1.0
    resolution: QtCore.QSize = deep_field(QtCore.QSize(64, 64))
    threshold: float = 0.2
    amount: float = 1.0
    mask_only: bool = False
    halation_only: bool = False


@hashable_dataclass
class Grain:
    samples: int = 1024
    blur_sigma: float = 0.5
    seed_offset: int = 0
    grain_mu: float = 0.07
    grain_sigma: float = 0.0
    lift: float = 0.000


@hashable_dataclass
class Render:
    # renderer
    force_resolution: bool = True
    resolution: QtCore.QSize = deep_field(QtCore.QSize(1920, 1080))

    # system
    device: str = ''


@hashable_dataclass
class Output:
    write: bool = False
    element: RenderElement = RenderElement.GRAIN
    path: str = '/home/beat/Downloads/ferrari_hamilton_live_v02_grain_0.07.exr'
    colorspace: str = 'ACES - ACEScg'
    frame: int = 0


@hashable_dataclass
class Project:
    input: Input = field(default_factory=Input)
    emulsion: Emulsion = field(default_factory=Emulsion)
    halation: Halation = field(default_factory=Halation)
    grain: Grain = field(default_factory=Grain)
    render: Render = field(default_factory=Render)
    output: Output = field(default_factory=Output)

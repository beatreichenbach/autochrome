import enum
from dataclasses import field

from PySide2 import QtCore

from qt_extensions.typeutils import hashable_dataclass, deep_field

from autochrome.api.tasks.opencl import Image


class EngineError(ValueError):
    pass


@enum.unique
class RenderElement(enum.Enum):
    GRAIN = enum.auto()
    SPECTRAL = enum.auto()
    GGX = enum.auto()
    HALATION = enum.auto()


@hashable_dataclass
class RenderImage:
    image: Image
    element: RenderElement


@hashable_dataclass
class Input:
    image_path: str = '/home/beat/dev/autochrome/data/DSC03544.exr'


@hashable_dataclass
class Grain:
    samples: int = 1024
    grain_mu: float = 0.02
    grain_sigma: float = 0.0
    blur_sigma: float = 0.5
    seed_offset: int = 0
    bounds_min: QtCore.QPoint = deep_field(QtCore.QPoint(0, 0))
    bounds_max: QtCore.QPoint = deep_field(QtCore.QPoint(1363, 2048))
    lift: float = 0.000


@hashable_dataclass
class Spectral:
    wavelength: int = 560


@hashable_dataclass
class GGX:
    roughness: float = 0.2
    height: float = 1.0
    light_position: QtCore.QPointF = deep_field(QtCore.QPointF(0.5, 0.5))
    resolution: QtCore.QSize = deep_field(QtCore.QSize(256, 256))
    mask: QtCore.QPointF = deep_field(QtCore.QPointF(0.1, 0.2))
    amount: float = 0.5


@hashable_dataclass
class Output:
    write: bool = False
    element: RenderElement = RenderElement.SPECTRAL
    path: str = '/home/beat/dev/autochrome/output_portra_green.exr'
    colorspace: str = 'ACEScg'
    frame: int = 0


@hashable_dataclass
class Render:
    # renderer
    resolution: QtCore.QSize = deep_field(QtCore.QSize(1920, 817))

    # system
    device: str = ''


@hashable_dataclass
class Project:
    input: Input = field(default_factory=Input)
    spectral: Spectral = field(default_factory=Spectral)
    grain: Grain = field(default_factory=Grain)
    ggx: GGX = field(default_factory=GGX)
    output: Output = field(default_factory=Output)
    render: Render = field(default_factory=Render)

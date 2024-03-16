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


@hashable_dataclass
class RenderImage:
    image: Image
    element: RenderElement


@hashable_dataclass
class Input:
    image_path: str = '/home/beat/Downloads/input_acescg.exr'


@hashable_dataclass
class Grain:
    samples: int = 64
    grain_mu: float = 0.05
    grain_sigma: float = 0.0
    blur_sigma: float = 0.4
    seed_offset: int = 0
    bounds_min: QtCore.QPointF = deep_field(QtCore.QPointF(0, 0))
    bounds_max: QtCore.QPointF = deep_field(QtCore.QPointF(2522, 1073))


@hashable_dataclass
class Spectral:
    wavelength: int = 560


@hashable_dataclass
class Output:
    write: bool = False
    element: RenderElement = RenderElement.GRAIN
    path: str = '/home/beat/dev/autochrome/output.exr'
    colorspace: str = 'ACEScg'
    frame: int = 0


@hashable_dataclass
class Render:
    # renderer
    resolution: QtCore.QSize = deep_field(QtCore.QSize(2522, 1073))

    # system
    device: str = ''


@hashable_dataclass
class Project:
    input: Input = field(default_factory=Input)
    spectral: Spectral = field(default_factory=Spectral)
    grain: Grain = field(default_factory=Grain)
    output: Output = field(default_factory=Output)
    render: Render = field(default_factory=Render)

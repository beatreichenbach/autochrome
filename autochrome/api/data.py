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
    image_path: str = '/home/beat/dev/autochrome/test.jpg'


@hashable_dataclass
class Spectral:
    wavelength: int = 560


@hashable_dataclass
class Output:
    pass


@hashable_dataclass
class Render:
    # renderer
    resolution: QtCore.QSize = deep_field(QtCore.QSize(16, 16))

    # system
    device: str = ''


@hashable_dataclass
class Project:
    input: Input = field(default_factory=Input)
    spectral: Spectral = field(default_factory=Spectral)
    output: Output = field(default_factory=Output)
    render: Render = field(default_factory=Render)

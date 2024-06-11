from __future__ import annotations

import dataclasses

from PySide2 import QtWidgets, QtCore

from autochrome.api.data import Project, RenderElement
from autochrome.utils import ocio
from qt_extensions.parameters import (
    EnumParameter,
    FloatParameter,
    IntParameter,
    ParameterBox,
    ParameterEditor,
    ParameterWidget,
    PathParameter,
    SizeParameter,
    StringParameter,
    MultiFloatParameter,
    BoolParameter,
)
from qt_extensions.typeutils import cast, basic


@dataclasses.dataclass
class Vector3F:
    x: float = 0
    y: float = 0
    z: float = 0

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __len__(self):
        return len((self.x, self.y, self.z))

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]


class Vector3FParameter(MultiFloatParameter):
    multi_count = 3
    value_changed: QtCore.Signal = QtCore.Signal(Vector3F)

    _value: Vector3F = Vector3F(0, 0, 0)
    _default: Vector3F = Vector3F(0, 0, 0)

    def set_value(self, value: Vector3F | list | tuple) -> None:
        super().set_value(value)

    def value(self) -> Vector3F:
        return super().value()

    def _cast_to_type(self, values: tuple[float, ...]) -> Vector3F:
        return Vector3F(*values[:3])

    def _cast_to_tuple(self, value: Vector3F) -> tuple[float, ...]:
        return tuple(value)


class ProjectEditor(ParameterEditor):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.forms = {}

        self._init()

        # init defaults
        default_config = Project()
        values = basic(default_config)
        self.set_values(values, attr='default')

    def _init(self) -> None:
        # input
        box = self.add_group('input')
        box.set_box_style(ParameterBox.BUTTON)
        input_group = box.form

        parm = PathParameter(name='image_path')
        input_group.add_parameter(parm)

        parm = StringParameter('colorspace')
        parm.set_menu(ocio.colorspace_names())
        parm.set_tooltip('Colorspace from the OCIO config.\nFor example: ACES - ACEScg')
        input_group.add_parameter(parm)

        # emulsion
        box = self.add_group('emulsion')
        box.set_box_style(ParameterBox.BUTTON)
        spectral_group = box.form

        parm = IntParameter(name='wavelength_count')
        parm.set_line_min(3)
        parm.set_slider_visible(False)
        spectral_group.add_parameter(parm)

        parm = IntParameter(name='model_resolution')
        parm.set_line_min(2)
        parm.set_slider_visible(False)
        spectral_group.add_parameter(parm)

        parm = IntParameter(name='lambda_count')
        parm.set_line_min(2)
        parm.set_slider_visible(False)
        spectral_group.add_parameter(parm)

        parm = PathParameter(name='curves_file')
        spectral_group.add_parameter(parm)

        parm = StringParameter(name='standard_illuminant')
        spectral_group.add_parameter(parm)

        parm = StringParameter(name='cmfs_variation')
        spectral_group.add_parameter(parm)

        # halation
        box = self.add_group('halation')
        box.set_box_style(ParameterBox.BUTTON)
        ggx_group = box.form

        parm = FloatParameter(name='roughness')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        ggx_group.add_parameter(parm)

        parm = FloatParameter(name='height')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        ggx_group.add_parameter(parm)

        parm = SizeParameter(name='resolution')
        parm.set_slider_visible(False)
        ggx_group.add_parameter(parm)

        parm = FloatParameter(name='threshold')
        ggx_group.add_parameter(parm)

        parm = FloatParameter(name='amount')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        ggx_group.add_parameter(parm)

        parm = BoolParameter(name='mask_only')
        ggx_group.add_parameter(parm)

        parm = BoolParameter(name='halation_only')
        ggx_group.add_parameter(parm)

        # grain
        box = self.add_group('grain')
        box.set_box_style(ParameterBox.BUTTON)
        grain_group = box.form

        box = grain_group.add_group('render')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        grain_render_group = box.form
        grain_render_group.create_hierarchy = False

        parm = IntParameter(name='samples')
        parm.set_line_min(0)
        parm.set_slider_visible(False)
        grain_render_group.add_parameter(parm)

        parm = FloatParameter(name='blur_sigma')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        grain_render_group.add_parameter(parm)

        box = grain_group.add_group('distribution')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        grain_distribution_group = box.form
        grain_distribution_group.create_hierarchy = False

        parm = IntParameter(name='seed_offset')
        parm.set_slider_visible(False)
        grain_distribution_group.add_parameter(parm)

        parm = FloatParameter(name='grain_mu')
        parm.set_line_min(0.0001)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        grain_distribution_group.add_parameter(parm)

        parm = FloatParameter(name='grain_sigma')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        grain_distribution_group.add_parameter(parm)

        parm = FloatParameter(name='lift')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(0.1)
        grain_distribution_group.add_parameter(parm)

        # render
        box = self.add_group('render')
        box.set_box_style(ParameterBox.BUTTON)
        render_group = box.form

        parm = BoolParameter('force_resolution')
        render_group.add_parameter(parm)

        parm = SizeParameter(name='resolution')
        parm.set_slider_visible(False)
        parm.set_keep_ratio(False)
        render_group.add_parameter(parm)

        parm = StringParameter('device')
        render_group.add_parameter(parm)

        # output
        box = self.add_group('output')
        box.set_box_style(ParameterBox.BUTTON)
        output_group = box.form

        # actions
        self.render_to_disk = QtWidgets.QAction('Render to Disk', self)
        button = QtWidgets.QToolButton()
        button.setDefaultAction(self.render_to_disk)
        output_group.add_widget(button, column=2)

        # parameters
        parm = EnumParameter('element')
        parm.set_enum(RenderElement)
        parm.set_tooltip('Output element')
        output_group.add_parameter(parm)

        parm = PathParameter('path')
        parm.set_method(PathParameter.SAVE_FILE)
        parm.set_tooltip(
            'Output image path. Use $F4 to replace frame numbers.\n'
            'For example: render.$F4.exr'
        )
        output_group.add_parameter(parm)

        parm = StringParameter('colorspace')
        parm.set_menu(ocio.colorspace_names())
        parm.set_tooltip('Colorspace from the OCIO config.\nFor example: ACES - ACEScg')
        output_group.add_parameter(parm)

    def project(self) -> Project:
        values = self.values()

        project = cast(Project, values)

        return project

    def set_project(self, project: Project) -> None:
        values = basic(project)

        self.blockSignals(True)
        self.set_values(values)
        self.blockSignals(False)
        self.parameter_changed.emit(ParameterWidget())

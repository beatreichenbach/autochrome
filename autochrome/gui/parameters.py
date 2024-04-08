from __future__ import annotations

import random

from PySide2 import QtWidgets, QtCore

from autochrome.api.data import Project, RenderElement
from autochrome.utils import ocio
from qt_extensions import helper
from qt_extensions.parameters import (
    BoolParameter,
    EnumParameter,
    FloatParameter,
    IntParameter,
    ParameterBox,
    ParameterEditor,
    ParameterWidget,
    PathParameter,
    PointFParameter,
    PointParameter,
    SizeFParameter,
    SizeParameter,
    StringParameter,
    TabDataParameter,
)
from qt_extensions.typeutils import cast, basic

# from autochrome.storage import Storage

# storage = Storage()


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
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        input_group = box.form

        parm = PathParameter(name='image_path')
        input_group.add_parameter(parm)

        # spectral
        box = self.add_group('spectral')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        spectral_group = box.form

        parm = IntParameter(name='wavelength')
        parm.set_line_min(380)
        parm.set_line_max(780)
        parm.set_slider_min(380)
        parm.set_slider_max(780)
        spectral_group.add_parameter(parm)

        # grain
        box = self.add_group('grain')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        grain_group = box.form

        parm = IntParameter(name='samples')
        parm.set_line_min(0)
        parm.set_slider_visible(False)
        grain_group.add_parameter(parm)

        parm = IntParameter(name='seed_offset')
        parm.set_slider_visible(False)
        grain_group.add_parameter(parm)

        parm = FloatParameter(name='grain_mu')
        parm.set_line_min(0.0001)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        grain_group.add_parameter(parm)

        parm = FloatParameter(name='grain_sigma')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        grain_group.add_parameter(parm)

        parm = FloatParameter(name='blur_sigma')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        grain_group.add_parameter(parm)

        parm = PointParameter(name='bounds_min')
        grain_group.add_parameter(parm)

        parm = PointParameter(name='bounds_max')
        grain_group.add_parameter(parm)

        # ggx

        box = self.add_group('ggx')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
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

        parm = PointFParameter(name='light_position')
        ggx_group.add_parameter(parm)

        parm = SizeParameter(name='resolution')
        parm.set_slider_visible(False)
        parm.set_keep_ratio(False)
        ggx_group.add_parameter(parm)

        # render
        box = self.add_group('render')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        render_group = box.form

        parm = SizeParameter(name='resolution')
        parm.set_slider_visible(False)
        parm.set_keep_ratio(False)
        render_group.add_parameter(parm)

        # output
        box = self.add_group('output')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
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

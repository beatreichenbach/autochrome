from __future__ import annotations

import random

from PySide2 import QtWidgets, QtCore

from autochrome.api.data import Project
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
        input_group = box.form

        parm = IntParameter(name='wavelength')
        parm.set_line_min(380)
        parm.set_line_max(780)
        parm.set_slider_min(380)
        parm.set_slider_max(780)
        input_group.add_parameter(parm)

        # grain
        box = self.add_group('grain')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        input_group = box.form

        parm = IntParameter(name='samples')
        parm.set_line_min(0)
        parm.set_slider_visible(False)
        input_group.add_parameter(parm)

        parm = IntParameter(name='seed_offset')
        parm.set_slider_visible(False)
        input_group.add_parameter(parm)

        parm = FloatParameter(name='grain_mu')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        input_group.add_parameter(parm)

        parm = FloatParameter(name='grain_sigma')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        input_group.add_parameter(parm)

        parm = FloatParameter(name='blur_sigma')
        parm.set_line_min(0)
        parm.set_slider_min(0)
        parm.set_slider_max(1)
        input_group.add_parameter(parm)

        parm = PointFParameter(name='bounds_min')
        input_group.add_parameter(parm)

        parm = PointFParameter(name='bounds_max')
        input_group.add_parameter(parm)

        # render
        box = self.add_group('render')
        box.set_box_style(ParameterBox.SIMPLE)
        box.set_collapsible(False)
        output_group = box.form

        parm = SizeParameter(name='resolution')
        parm.set_slider_visible(False)
        parm.set_keep_ratio(False)
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

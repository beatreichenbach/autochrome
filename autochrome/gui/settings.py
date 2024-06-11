from __future__ import annotations

import logging

from PySide2 import QtWidgets, QtGui, QtCore

from autochrome.storage import Storage, Settings
from autochrome.utils import ocio
from qt_extensions.button import Button
from qt_extensions.messagebox import MessageBox
from qt_extensions.parameters import (
    BoolParameter,
    ParameterBox,
    PathParameter,
    ParameterEditor,
    StringParameter,
)
from qt_extensions.typeutils import basic, cast

logger = logging.getLogger(__name__)
storage = Storage()


class SettingsEditor(ParameterEditor):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._init_editor()

    def _init_editor(self) -> None:
        # color
        box = self.add_group('Color')
        box.set_box_style(ParameterBox.BUTTON)
        form = box.form
        form.create_hierarchy = False

        parm = PathParameter('ocio')
        parm.set_label('OCIO Config')
        parm.set_method(PathParameter.SAVE_FILE)
        parm.set_tooltip(
            'Path to the config.ocio file. '
            'Currently a ACES config is required. '
            'If no path is set here, the system will fall back to '
            'the environment variable \'OCIO\''
        )
        form.add_parameter(parm)

        parm = StringParameter('view_colorspace')
        parm.set_menu(ocio.view_names())
        form.add_parameter(parm)

        # logging
        box = self.add_group('Logging')
        box.set_box_style(ParameterBox.BUTTON)
        form = box.form
        form.create_hierarchy = False

        parm = BoolParameter('clear_log_on_render')
        parm.set_tooltip('Clear the log on every render.')
        form.add_parameter(parm)

        # crash reporting
        box = self.add_group('Crash Reporting')
        box.set_box_style(ParameterBox.BUTTON)
        form = box.form
        form.create_hierarchy = False

        parm = BoolParameter('sentry')
        parm.set_label('Automated Crash Reporting')
        parm.set_tooltip(
            'Automatically upload crash reports using Sentry.io. '
            'Crash reports don\'t include any personal information.'
        )
        form.add_parameter(parm)

    def settings(self) -> Settings:
        values = self.values()
        settings = cast(Settings, values)
        return settings

    def set_settings(self, settings: Settings, attr: str = 'value') -> None:
        settings_dict = basic(settings)
        if settings_dict['sentry'] is None:
            settings_dict['sentry'] = True
        self.blockSignals(True)
        self.set_values(settings_dict, attr=attr)
        self.blockSignals(False)


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._init_ui()
        self.setWindowTitle('Settings')
        self.resize(QtCore.QSize(800, 600))

    def _init_ui(self) -> None:
        self.setLayout(QtWidgets.QVBoxLayout())

        # editor
        self.editor = SettingsEditor()
        self.editor.set_settings(storage.settings, attr='default')
        self.editor.parameter_changed.connect(self._settings_changed)

        self.layout().addWidget(self.editor)

        self.button_box = QtWidgets.QDialogButtonBox()
        size_policy = self.button_box.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.button_box.setSizePolicy(size_policy)
        self.layout().addWidget(self.button_box)

        save_button = Button('Save', color='primary')
        save_button.pressed.connect(self.save)
        self.button_box.addButton(save_button, QtWidgets.QDialogButtonBox.ApplyRole)

        cancel_button = Button('Cancel')
        cancel_button.pressed.connect(self.cancel)
        self.button_box.addButton(cancel_button, QtWidgets.QDialogButtonBox.RejectRole)

        self.button_box.hide()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.check_save():
            super().closeEvent(event)
        else:
            event.ignore()

    def cancel(self) -> None:
        self.editor.set_settings(storage.settings)
        self.button_box.hide()

    def check_save(self) -> bool:
        """Return True if program can continue, False if action should be cancelled."""

        settings = self.editor.settings()
        if storage.settings == settings:
            return True

        buttons = (
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Cancel
            | QtWidgets.QMessageBox.StandardButton.Discard
        )

        result = MessageBox.question(
            parent=self,
            title='Unsaved Changes',
            text='You have unsaved changes that will be lost. Do you want to save them?',
            buttons=buttons,
        )

        if result == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        elif result == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        else:
            return True

    def save(self) -> bool:
        storage.settings = self.editor.settings()
        storage.update_ocio()
        result = storage.save_settings()
        if result:
            self.button_box.hide()
        return result

    def _settings_changed(self) -> None:
        settings = self.editor.settings()
        if storage.settings != settings:
            self.button_box.show()
        else:
            self.button_box.hide()

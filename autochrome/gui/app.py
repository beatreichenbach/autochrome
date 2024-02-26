from __future__ import annotations

import copy
import inspect
import logging
import os
import sys
from functools import partial
from importlib.metadata import version
from importlib.resources import files

import pyopencl as cl
from PySide2 import QtCore, QtGui, QtWidgets
from autochrome.api.data import Project, RenderImage
from autochrome.api.engine import Engine, clear_cache
from autochrome.gui.parameters import ProjectEditor
from autochrome.gui.settings import SettingsDialog
from autochrome.gui.viewer import ElementViewer
from autochrome.storage import Storage

from qt_extensions import theme
from qt_extensions.icons import MaterialIcon
from qt_extensions.logger import LogCache, LogBar, LogViewer
from qt_extensions.mainwindow import DockWindow, DockWidgetState, SplitterState

logger = logging.getLogger(__name__)
storage = Storage()


class MainWindow(DockWindow):
    render_requested: QtCore.Signal = QtCore.Signal(Project)
    stop_requested: QtCore.Signal = QtCore.Signal()
    elements_changed: QtCore.Signal = QtCore.Signal(list)

    default_window_state = {
        'widgets': [
            SplitterState(
                sizes=[1, 1],
                orientation=QtCore.Qt.Horizontal,
                states=[
                    DockWidgetState(
                        current_index=0,
                        widgets=[('Viewer 1', ElementViewer.__name__)],
                        detachable=True,
                        auto_delete=False,
                        is_center_widget=True,
                    ),
                    DockWidgetState(
                        current_index=0,
                        widgets=[('Parameters', ProjectEditor.__name__)],
                        detachable=True,
                        auto_delete=True,
                        is_center_widget=False,
                    ),
                ],
            )
        ]
    }

    default_widget_states = {'Log': {'names': ['autochrome']}}

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._init_splash_screen()

        self.project = Project()
        self.rendering = False

        self._api_thread = QtCore.QThread()
        self._project_queue = None
        self._project_path = ''
        self._project_hash = hash(self.project)
        self._project_changed = False
        self._device = ''
        self._recent_dir = os.path.expanduser('~')

        self.show_splash_message('Loading User Interface...')
        self._init_log()
        self._init_widgets()
        self.show_splash_message('Loading Menu...')
        self._init_menu()
        self.show_splash_message('Loading State...')
        self.load_state()
        self.show_splash_message('Loading Project...')
        if storage.state.recent_paths:
            self.file_open(storage.state.recent_paths[0])
        else:
            self.update_window_title()
        self.show_splash_message('Loading Engine...')
        self._init_engine()

    def _init_splash_screen(self) -> None:
        filename = os.path.join(os.path.dirname(__file__), 'assets', 'splash.png')
        pixmap = QtGui.QPixmap(str(filename))
        size = QtCore.QSize(800, 500)
        pixmap = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio)
        self.splash_screen = QtWidgets.QSplashScreen(pixmap)
        self.splash_screen.show()

    def _init_log(self) -> None:
        # logging
        self.log_cache = LogCache()
        self.log_cache.connect_logger(logging.getLogger())

        # status bar
        self.log_bar = LogBar(self.log_cache)
        self.log_bar.open_viewer = lambda: self.show_widget(LogViewer)
        self.log_bar.names = ['root', 'autochrome']
        self.log_bar.level = logging.WARNING
        self.layout().addWidget(self.log_bar)

        # progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.log_bar.add_widget(self.progress_bar)

    def _init_engine(self) -> None:
        self._api_thread = QtCore.QThread()
        self.rendering = False

        try:
            self.engine = Engine(self.project.render.device)
        except (cl.Error, ValueError) as e:
            self.engine = None
            logger.error(e)
            logger.error('failed to start the engine')
            return

        self.engine.moveToThread(self._api_thread)
        self._api_thread.start()

        self.engine.image_rendered.connect(self._image_rendered)
        self.engine.progress_changed.connect(self._progress_changed)
        self.elements_changed.connect(self.engine.set_elements)
        self.render_requested.connect(self.engine.render)

    def _init_menu(self) -> None:
        menu_bar = QtWidgets.QMenuBar(self)
        self.layout().insertWidget(0, menu_bar)

        # file
        file_menu = menu_bar.addMenu('File')

        # action = QtWidgets.QAction('New', self)
        # action.setShortcut(QtGui.QKeySequence.New)
        # action.triggered.connect(self.file_new)
        # file_menu.addAction(action)
        # action = QtWidgets.QAction('Open...', self)
        # action.setIcon(MaterialIcon('file_open'))
        # action.setShortcut(QtGui.QKeySequence.Open)
        # action.triggered.connect(lambda: self.file_open())
        # file_menu.addAction(action)
        self.recent_menu = file_menu.addMenu('Open Recent...')
        file_menu.addSeparator()

        action = QtWidgets.QAction('Settings...', self)
        action.setShortcut(QtGui.QKeySequence('Ctrl+Alt+S'))
        action.setIcon(MaterialIcon('settings'))
        action.triggered.connect(lambda: self.settings_open())
        file_menu.addAction(action)
        file_menu.addSeparator()

        # action = QtWidgets.QAction('Save', self)
        # action.setIcon(MaterialIcon('save'))
        # action.setShortcut(QtGui.QKeySequence.Save)
        # action.triggered.connect(self.file_save)
        # file_menu.addAction(action)
        # action = QtWidgets.QAction('Save As...', self)
        # action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+S'))
        # action.triggered.connect(self.file_save_as)
        # file_menu.addAction(action)
        # file_menu.addSeparator()

        action = QtWidgets.QAction('Exit', self)
        action.setShortcut(QtGui.QKeySequence.Quit)
        action.triggered.connect(self.close)
        file_menu.addAction(action)

        # view
        view_menu = menu_bar.addMenu('View')
        action = QtWidgets.QAction('New Viewer', self)
        # action.setIcon(MaterialIcon('preview'))
        action.triggered.connect(partial(self.show_widget, ElementViewer))
        view_menu.addAction(action)
        action = QtWidgets.QAction('Show Parameters', self)
        # action.setIcon(MaterialIcon('tune'))
        action.triggered.connect(partial(self.show_widget, ProjectEditor))
        view_menu.addAction(action)
        view_menu.addSeparator()
        action = QtWidgets.QAction('Reset', self)
        action.triggered.connect(self.reset_window_state)
        view_menu.addAction(action)

        # engine
        view_menu = menu_bar.addMenu('Engine')
        action = QtWidgets.QAction('Restart', self)
        # action.setIcon(MaterialIcon('restart_alt'))
        action.triggered.connect(self.restart)
        view_menu.addAction(action)

        # help
        # help_menu = menu_bar.addMenu('Help')
        # action = QtWidgets.QAction('Documentation', self)
        # action.setIcon(MaterialIcon('question_mark'))
        # action.triggered.connect(self.help_documentation)
        # help_menu.addAction(action)
        # action = QtWidgets.QAction('Report an Issue', self)
        # action.setIcon(MaterialIcon('bug_report'))
        # action.triggered.connect(self.help_report_bug)
        # help_menu.addAction(action)
        # help_menu.addSeparator()
        # action = QtWidgets.QAction('Check for Updates', self)
        # action.setIcon(MaterialIcon('update'))
        # action.triggered.connect(self.help_update)
        # help_menu.addAction(action)
        # action = QtWidgets.QAction('About', self)
        # action.triggered.connect(self.help_about)
        # help_menu.addAction(action)

    def _init_widgets(self) -> None:
        self.register_widget(ElementViewer, 'Viewer', unique=False)
        self.register_widget(ProjectEditor, 'Parameters')
        self.register_widget(LogViewer, 'Log')

        # connect signals when new widgets are added
        self.widget_added.connect(self._widget_added)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # project
        self._test_changes(quick=False)
        if self._project_changed:
            result = QtWidgets.QMessageBox.warning(
                self,
                'Save Changes?',
                'Project has been modified, save changes?',
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel,
            )
            if result == QtWidgets.QMessageBox.Yes:
                # if not self.file_save():
                #     event.ignore()
                #     return
                pass
            elif result == QtWidgets.QMessageBox.No:
                pass
            else:
                event.ignore()
                return

        # engine
        self._api_thread.quit()

        # state
        self.save_state()

        super().closeEvent(event)

    def load_state(self) -> None:
        # window state
        if storage.state.window_state:
            self.set_window_state(storage.state.window_state)
        else:
            self.reset_window_state()

        # widget states
        for title, state in storage.state.widget_states.items():
            widget = self._widgets.get(title)
            set_widget_state(widget, state)

        # menu
        self._update_recent_menu()

    def refresh(self) -> None:
        self._update_elements()
        self.request_render(self.project)

    def render_to_disk(self) -> None:
        self.elements_changed.emit([self.project.output.element])
        self.project.output.write = True
        self.request_render(self.project)
        self.project.output.write = False
        self._update_elements()

    def reset_window_state(self) -> None:
        self.set_window_state(self.default_window_state)

    def restart(self):
        self._api_thread.quit()
        clear_cache()
        self._init_engine()
        self.refresh()

    def request_render(
        self,
        project: Project | None = None,
    ) -> None:
        # try starting a render, if the engine is rendering,
        # store the project in the queue instead
        if project is not None:
            # TODO: so far the only reason for deepcopy is render_to_disk.
            #  If that function wouldn't require setting a temporary setting
            #  it could stay mutable...

            self._project_queue = copy.deepcopy(project)

        if self.rendering:
            self.stop_requested.emit()
        elif self._project_queue is not None and self._api_thread.isRunning():
            self.rendering = True
            self.render_requested.emit(self._project_queue)
            self._project_queue = None

    def save_state(self) -> None:
        storage.state.window_state = self.state()

        for title, widget in self._widgets.items():
            state = widget_state(widget)
            if state is not None:
                storage.state.widget_states[title] = state

        storage.save_state()

    def set_project(self, project: Project) -> None:
        self.project = project
        self._project_hash = hash(self.project)
        for title, widget in self._widgets.items():
            if isinstance(widget, ProjectEditor):
                widget.set_project(self.project)

    def settings_open(self) -> None:
        dialog = SettingsDialog(parent=self)
        dialog.open()

    def show_splash_message(self, message: str) -> None:
        if isinstance(self.splash_screen, QtWidgets.QSplashScreen):
            self.splash_screen.showMessage(
                message,
                int(QtCore.Qt.AlignBottom or QtCore.Qt.AlignLeft),
                QtGui.QColor('white'),
            )

    def show_widget(self, cls: type[QtWidgets.QWidget]) -> None:
        try:
            dock_widget = self.create_dock_widget(cls)
            dock_widget.float()
            dock_widget.show()
        except ValueError:
            # widget already exists
            for widget in self._widgets.values():
                if isinstance(widget, cls):
                    self.focus_widget(widget)
                    break

    def update_window_title(self) -> None:
        filename = os.path.basename(self._project_path)
        title = filename if filename else 'untitled'
        if self._project_changed:
            title = f'{title} *'
        self.setWindowTitle(title)

    def _update_elements(self) -> None:
        elements = []
        for dock_widget in self.dock_widgets():
            widget = dock_widget.currentWidget()
            if isinstance(widget, ElementViewer) and not widget.paused:
                elements.append(widget.element)
        self.elements_changed.emit(elements)

    def _update_recent_menu(self) -> None:
        self.recent_menu.clear()
        for filename in storage.state.recent_paths:
            action = QtWidgets.QAction(filename, self)
            action.triggered.connect(partial(self.file_open, filename))
            self.recent_menu.addAction(action)

    def _widget_added(self, widget: QtWidgets.QWidget) -> None:
        # restore state
        title = self.widget_title(widget)
        if title is not None:
            states = self.default_widget_states.copy()
            states.update(storage.state.widget_states)
            state = states.get(title)
            set_widget_state(widget, state)

        # reconnect signals
        if isinstance(widget, ElementViewer):
            widget.element_changed.connect(lambda: self._viewer_changed())
            widget.pause_changed.connect(lambda: self._viewer_changed())
            widget.refreshed.connect(self.refresh)
        elif isinstance(widget, ProjectEditor):
            widget.set_project(self.project)
            widget.parameter_changed.connect(
                lambda: self._project_editor_changed(widget)
            )
            widget.render_to_disk.triggered.connect(self.render_to_disk)
        elif isinstance(widget, LogViewer):
            widget.set_cache(self.log_cache)

    def _viewer_changed(self) -> None:
        self._update_elements()
        self.request_render(self.project)

    def _project_editor_changed(self, editor: ProjectEditor) -> None:
        try:
            self.project = editor.project()
        except TypeError as e:
            logger.exception(e)
            self.project = None
        self._test_changes()

        # device
        if self.project.render.device != self._device:
            self._device = self.project.render.device
            self.restart()

        # render
        self.request_render(self.project)

    def _progress_changed(self, value: float) -> None:
        if value == 0:
            if storage.settings.clear_log_on_render:
                self.log_cache.clear()
            self.progress_bar.setMaximum(0)
        if value >= 1:
            self.rendering = False
            self.progress_bar.setMaximum(100)
            self.request_render()

    def _image_rendered(self, image: RenderImage) -> None:
        for widget in self._widgets.values():
            if isinstance(widget, ElementViewer) and widget.element == image.element:
                widget.set_array(image.image.array)

    def _test_changes(self, quick: bool = True) -> None:
        # checks whether project has changed
        if quick and self._project_changed:
            # don't perform hash comparisons for performance
            return

        self._project_changed = hash(self.project) != self._project_hash
        self.update_window_title()

    def _update_elements(self) -> None:
        elements = []
        for dock_widget in self.dock_widgets():
            widget = dock_widget.currentWidget()
            if isinstance(widget, ElementViewer) and not widget.paused:
                elements.append(widget.element)
        self.elements_changed.emit(elements)

    def _viewer_changed(self) -> None:
        self._update_elements()
        self.request_render(self.project)

    def _widget_added(self, widget: QtWidgets.QWidget) -> None:
        # restore state
        title = self.widget_title(widget)
        if title is not None:
            states = self.default_widget_states.copy()
            states.update(storage.state.widget_states)
            state = states.get(title)
            set_widget_state(widget, state)

        # reconnect signals
        if isinstance(widget, ElementViewer):
            widget.element_changed.connect(lambda: self._viewer_changed())
            widget.pause_changed.connect(lambda: self._viewer_changed())
            widget.refreshed.connect(self.refresh)
        elif isinstance(widget, ProjectEditor):
            widget.set_project(self.project)
            widget.parameter_changed.connect(
                lambda: self._project_editor_changed(widget)
            )
            widget.render_to_disk.triggered.connect(self.render_to_disk)
        elif isinstance(widget, LogViewer):
            widget.set_cache(self.log_cache)


def set_widget_state(widget: QtWidgets.QWidget, state: dict | None) -> None:
    if state is None:
        return
    if hasattr(widget, 'set_state') and inspect.ismethod(widget.set_state):
        widget.set_state(state)


def widget_state(widget: QtWidgets.QWidget) -> dict | None:
    if hasattr(widget, 'state') and inspect.ismethod(widget.state):
        return widget.state()


def init_app() -> QtWidgets.QApplication:
    # application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('autochrome')
    app.setApplicationDisplayName('Autochrome')
    app.setApplicationVersion(version('autochrome'))
    icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.png')
    icon = QtGui.QIcon(str(icon_path))
    app.setWindowIcon(icon)

    # theme
    theme.apply_theme(theme.monokai)

    return app


def init_window(project: str = '') -> MainWindow:
    # sentry

    if project:
        storage.add_recent_path(project)

    # main window
    window = MainWindow()
    window.show()

    # render
    window.refresh()

    return window


def exec_():
    app = init_app()
    init_window()
    return app.exec_()


if __name__ == '__main__':
    logging.basicConfig()
    os.environ['OPENCL_REBUILD'] = '1'
    os.environ['OCIO'] = (
        '/home/beat/Downloads/cg-config-v2.1.0_aces-v1.3_ocio-v2.3.ocio'
    )
    os.environ['OCIO_INACTIVE_COLORSPACES'] = ''
    exec_()

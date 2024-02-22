import logging
import sys

from PySide2 import QtWidgets
from qt_extensions import theme

from autochrome.gui.viewer import ElementViewer
from qt_extensions.viewer import Viewer


def main():
    logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(theme.monokai)

    viewer = Viewer()
    viewer.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

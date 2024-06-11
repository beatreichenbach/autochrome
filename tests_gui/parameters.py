# import json
import logging
import sys

from PySide2 import QtWidgets
from qt_extensions import theme

# from autochrome.api import data
from autochrome.gui.parameters import ProjectEditor


# from qt_extensions.typeutils import basic, cast


def main():
    logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(theme.monokai)

    editor = ProjectEditor()

    editor.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

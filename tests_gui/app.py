import logging
import os

from autochrome.gui import app


if __name__ == '__main__':
    logging.basicConfig()

    logging.getLogger('pyopencl').setLevel(logging.WARNING)
    logging.getLogger('pytools').setLevel(logging.WARNING)

    os.environ['OPENCL_REBUILD'] = '1'
    os.environ['OCIO_INACTIVE_COLORSPACES'] = ''

    app.exec_()

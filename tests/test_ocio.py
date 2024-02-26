import json
import logging
import os
import sys

from autochrome.utils import ocio

logger = logging.getLogger(__name__)


def test_colorspace_names():
    names = ocio.colorspace_names()
    logger.info(json.dumps(names, indent=2))


def test_view_names():
    names = ocio.view_names()
    logger.info(json.dumps(names, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    os.environ['OCIO_INACTIVE_COLORSPACES'] = ''
    test_colorspace_names()
    test_view_names()

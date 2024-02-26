from __future__ import annotations

import logging
from functools import lru_cache

import PyOpenColorIO as OCIO

from autochrome.storage import Storage

logger = logging.getLogger(__name__)
storage = Storage()
storage.update_ocio()

GROUP_DELIMITER = ' - '
VIEW_DELIMITER = ': '
RENDER_SPACE = 'ACES - ACEScg'


def colorspace_names() -> dict:
    names = {}

    try:
        config = OCIO.GetCurrentConfig()
    except OCIO.Exception:
        return names

    for name in config.getColorSpaceNames():
        logger.debug(f'colorspace_name: {name}')
        max_split = 1
        if name.startswith('Input'):
            max_split = 2
        words = name.split(GROUP_DELIMITER, max_split)

        group_name = words[0]
        group = names.get(group_name, {})

        if isinstance(group, dict):
            if len(words) == 1:
                names[name] = name
            elif len(words) == 2:
                group[words[1]] = name
                names[group_name] = group
            elif len(words) == 3:
                sub_group_name = words[1]
                sub_group = group.get(sub_group_name, {})
                if isinstance(sub_group, dict):
                    sub_group[words[2]] = name
                    group[sub_group_name] = sub_group
                    names[group_name] = group
    return names


def view_names() -> dict:
    names = {}

    try:
        config = OCIO.GetCurrentConfig()
    except OCIO.Exception:
        return names

    for display in config.getDisplays():
        views = names.get(display, {})
        for view in config.getViews(display):
            views[view] = f'{display}{VIEW_DELIMITER}{view}'
            names[display] = views

    return names


@lru_cache(10)
def view_processor() -> OCIO.Processor | None:
    colorspace = storage.settings.view_colorspace
    if not storage.settings.view_colorspace:
        return

    words = colorspace.split(VIEW_DELIMITER)
    if len(words) != 2:
        logger.warning(f'Invalid view color space: {colorspace}')
        return

    display, view = words
    direction = OCIO.TransformDirection.TRANSFORM_DIR_FORWARD

    try:
        config = OCIO.GetCurrentConfig()
        processor = config.getProcessor(RENDER_SPACE, display, view, direction)
        cpu_processor = processor.getDefaultCPUProcessor()
        logger.info(
            f'Processor created: CPUProcessor('
            f'working_space={RENDER_SPACE}, display_space={display}, view={view})'
        )
        return cpu_processor
    except OCIO.Exception as e:
        logger.debug(e)
        logger.warning('Failed to initialize OCIO.')


@lru_cache(10)
def colorspace_processor(
    src_name: str = RENDER_SPACE, dst_name: str = RENDER_SPACE
) -> OCIO.CPUProcessor | None:
    if src_name == dst_name:
        return

    try:
        config = OCIO.GetCurrentConfig()
        src_colorspace = config.getColorSpace(src_name)
        dst_colorspace = config.getColorSpace(dst_name)
        processor = config.getProcessor(src_colorspace, dst_colorspace)
        cpu_processor = processor.getDefaultCPUProcessor()
        logger.info(
            f'Processor created: CPUProcessor('
            f'src_name={src_name}, dst_name={dst_name})'
        )
        return cpu_processor
    except OCIO.Exception as e:
        logger.debug(e)
        logger.warning('Failed to initialize OCIO.')

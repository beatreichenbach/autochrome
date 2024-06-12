import argparse
import logging
import sys

from autochrome.cli import app as cli_app
from autochrome.gui import app as gui_app


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='Autochrome',
        description='Realistic Film Grain',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='run the application in gui mode',
    )
    parser.add_argument(
        '--project',
        type=str,
        help='the project to render, a path to a .json file',
    )
    parser.add_argument(
        '--animation',
        type=str,
        help='path to a .json file containing animation data',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output image',
    )
    parser.add_argument(
        '--element',
        type=str,
        help='element to render, for example: '
        'GRAIN, GGX ... HALATION',
    )
    parser.add_argument(
        '--colorspace',
        type=str,
        default='ACES - ACEScg',
        help='output colorspace',
    )
    parser.add_argument(
        '--frame-start',
        type=int,
        default=1,
        help='start frame number',
    )
    parser.add_argument(
        '--frame-end',
        type=int,
        default=1,
        help='end frame number',
    )
    parser.add_argument(
        '--log',
        type=int,
        default=logging.WARNING,
        help='logging level',
    )
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args(sys.argv[1:])

    if args.gui:
        gui_app.exec_()
    else:
        try:
            cli_app.exec_(args)
        except Exception as e:
            parser.error(str(e))


if __name__ == '__main__':
    main()

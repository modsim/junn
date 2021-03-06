import argparse
import sys

from tunable import TunableSelectable

from .logging import get_log_levels, setup_logging

HELP = '--help'


def get_common_argparser_and_setup(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', dest='tag', type=str, help='tag')

    TunableSelectable.setup_and_parse(parser, args=args)

    parser.add_argument(
        'input', metavar='input', type=str, nargs='*', help="input file(s)"
    )
    parser.add_argument(
        '--log-level',
        dest='log_level',
        type=str,
        choices=get_log_levels(),
        default='INFO',
    )
    parser.add_argument(
        '--model', dest='model', type=str, help="model path (TensorFlow format)"
    )

    help_removed = False

    if HELP in args:
        args.remove(HELP)
        help_removed = True

    args_parsed, _ = parser.parse_known_args(args=args)

    setup_logging(args_parsed.log_level)

    if hasattr(args_parsed, 'NeuralNetwork') and args_parsed.NeuralNetwork:
        args = [
            arg.replace('%NeuralNetwork', args_parsed.NeuralNetwork) for arg in args
        ]

    args_parsed, _ = parser.parse_known_args(args=args)

    if args_parsed.tag:
        args = [arg.replace('%tag', args_parsed.tag) for arg in args]

    if help_removed:
        args.append(HELP)

    return args, parser

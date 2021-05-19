#!/usr/bin/env python

import glob
import os
import os.path as osp
import sys

# noinspection PyProtectedMember
from runpy import _run_module_as_main

if os.path.isdir('junn-predict'):
    sys.path.insert(0, 'junn-predict')

from junn_predict.common.configure_tensorflow import get_gpu_memory_usages_megabytes

base_name = 'junn'
memory_usage_warning_threshold = 0.9


def get_possible_commands():
    base_directory = osp.dirname(osp.dirname(osp.realpath(__file__)))

    base_directory = osp.dirname(base_directory)

    def strip_name(name):
        return (
            name.replace(base_directory + osp.sep, '')
            .replace(base_name + osp.sep, '')
            .replace(osp.sep + '__main__.py', '')
            .replace(osp.sep, '.')
        )

    raw_commands = glob.glob(base_directory + '/**/__main__.py', recursive=True)
    commands = [strip_name(module) for module in raw_commands if module != __file__]
    commands = [module for module in commands if module != '__main__.py']

    return commands


def print_possible_commands():
    print("Possible commands:")
    print()

    for module in sorted(get_possible_commands()):
        print(
            '%s %s'
            % (
                base_name,
                module,
            )
        )


def get_gpu_memory_usage_fractions():
    return [
        used / total if total else 0.0
        for used, total in get_gpu_memory_usages_megabytes()
    ]


def get_mean_gpu_memory_usage_fraction():
    memory_usages = get_gpu_memory_usage_fractions()
    return sum(memory_usages) / len(memory_usages)


def main():
    """
    This is the launcher main(). It will look for callable (i.e. __main__.py containing) sub-modules and offer them
    as choices to run. If an argument is passed, it tries to call them. Furthermore it has two additional
    functions: If '-nogpu' is passed, it will disable GPU usage via CUDA_VISIBLE_DEVICES environment variable,
    and it will warn if the GPU is to be used, but its memory is more than 90% full (which likely means,
    another process is still using it).

    :return:
    """

    if len(sys.argv) == 1:
        return print_possible_commands()

    del sys.argv[0]

    if sys.argv[0] in ('-cpu', '-nogpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        del sys.argv[0]

    if (
        'CUDA_VISIBLE_DEVICES' not in os.environ
        and get_mean_gpu_memory_usage_fraction() > memory_usage_warning_threshold
    ):
        print(
            "More than %.2f%% of GPU memory used, likely some other process is still using the GPU!"
            % (memory_usage_warning_threshold * 100.0)
        )

    sys.argv[0] = base_name + '.' + sys.argv[0]

    _run_module_as_main(sys.argv[0])


if __name__ == '__main__':
    main()

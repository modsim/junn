import sys
from contextlib import contextmanager

import pytest

from junn.common.launcher.__main__ import main


@contextmanager
def alter_sys_argv(new_sys_argv=None):
    if new_sys_argv is None:
        new_sys_argv = ['script.py']
    real_sys_argv = sys.argv
    sys.argv = new_sys_argv
    try:
        yield True
    finally:
        sys.argv = real_sys_argv


def test_launcher():
    from importlib.machinery import SourceFileLoader

    with alter_sys_argv():
        SourceFileLoader('__main__', 'junn/common/launcher/__main__.py').load_module()


def test_launcher_nogpu():
    with alter_sys_argv(['-nogpu', 'train', '--help']):
        with pytest.raises(SystemExit):
            main()

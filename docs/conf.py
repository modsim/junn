# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Build like:
# sphinx-build -b html docs docs/_build

# -- Path setup --------------------------------------------------------------


import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.join(os.path.abspath('..'), 'junn-predict'))

import junn_predict  # noqa

import junn  # noqa

try:
    import sphinx_rtd_theme
except ImportError:
    sphinx_rtd_theme = None

autodoc_mock_imports = [
    'skimage',
    'tqdm',
    'cv2',
    'tensorflow',
    'tensorflow_addons',
    'keras',
]

# -- Project information -----------------------------------------------------

project = junn.__name__
copyright = junn.__copyright__
author = junn.__author__
release = junn.__version__

import sphinx

# -- General configuration ---------------------------------------------------

sys.path.insert(0, os.path.abspath('./_ext'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'automagicdoc']

automagic_modules = [junn, junn_predict]
automagic_ignore = ['*test*']

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


if sphinx_rtd_theme:
    extensions.append('sphinx_rtd_theme')
    html_theme = 'sphinx_rtd_theme'

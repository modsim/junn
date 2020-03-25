# -*- coding: utf-8 -*-
"""
documentation
"""

import os

from setuptools import setup, find_packages

package = os.environ.get('BUILD_PACKAGE')
if package is None:
    package = 'junn'

if package == 'junn':

    setup(
        name='junn',
        version='0.0.1',
        install_requires=[
            'numpy',
            'scipy',
            'jsonpickle',
            'tensorflow',
            'tensorflow-addons',
            'tensorflow-serving-api',
            'horovod',
            'scikit-image',
            # 'opencv',  # no proper way of requiring this pip-compatible
            'tunable',
            'py3nvml',
            'jsonpickle',
            'tqdm',
            'colorlog',
            'pillow',
            'requests',
            'pilyso-io',
            'tifffile',
            'roifile'
        ],
        packages=find_packages(include=('junn', 'junn.*'))
    )

elif package == 'junn-predict':

    setup(
        name='junn-predict',
        version='0.0.1',
        install_requires=[
            'numpy',
            # 'opencv',  # no proper way of requiring this pip-compatible
            'tunable',
            'tqdm',
            'pillow',
            'requests',
            'pilyso-io',
            'tifffile',
            'roifile'
        ],
        packages=find_packages(include=('junn_predict', 'junn_predict.*'))
    )

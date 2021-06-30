JUNN Readme
===========

**Important**: The repository and accompanying manuscript are currently in the process of being finalized. During this period, not everything in the repository might be completely finished or in complete working order yet.

The Jülich U-Net Neural Network Toolkit. JUNN is a neural network training tool, aimed at allowing the easy training of configurable U-Nets or other pixel to pixel segmentation networks, with the resulting networks usable in a standalone manner.

Citation
========

The manuscript for JUNN is currently in preparation, if you use JUNN, please cite the upcoming publication.

Prerequisites
=============

JUNN is split into two components: a training and a prediction component. The training component runs the computations locally, the usage of a CUDA-compatible GPU is highly advisable.
The prediction component can either run the predictions locally, or offload the computations to remote machines, and in general prediction is, albeit slower, still somewhat feasible with CPU-only processing speed. Via the remote prediction functionality, a server with GPUs may be shared for multiple non-GPU prediction clients, see the documentation how such a scenario may be set up.

Installation/Usage
==================

Docker
------

The simplest way to run JUNN is to use the Docker images provided (note that to use a GPU inside a container, the `necessary runtime <https://github.com/NVIDIA/nvidia-container-runtime>`_ needs to be installed) :

.. code-block:: bash

    # training
    > docker run --gpus all --ipc=host --tty --interactive --rm -v `pwd`:/data modsim/junn train --NeuralNetwork Unet --model model_name training_input.tif
    # prediction
    > docker run --gpus all --ipc=host --tty --interactive --rm -v `pwd`:/data modsim/junn predict --model model_name data.tif

Anaconda
--------

To install JUNN locally, we recommend using the Anaconda package manager.

Since Anaconda TensorFlow packages tend to lag a bit behind, it is necessary to install the official TensorFlow binaries from PyPI.

.. code-block:: bash

    > conda create -n junn
    > conda activate junn
    > conda install -c modsim junn
    > pip install tensorflow tensorflow-addons tensorflow-serving-api
    # usage
    > python -m junn train --NeuralNetwork Unet --model model_name training_input.tif
    > python -m junn predict --model model_name data.tif

Github
------

Installing JUNN from Github is only recommended for users familiar with Python development. Either conda packages can be built locally and installed, or the pip packages can be built and installed, or the necessary dependencies can be installed and the source run in-place.

.. code-block:: bash

    # let conda build and install the packages
    > conda create -n junn
    > conda activate junn
    > conda build junn-predict/recipe recipe
    > conda install -c local junn-predict junn

    # ... or let pip build and install the packages
    > pip install junn-predict
    > pip install .

    # always install TensorFlow via pip
    > pip install tensorflow tensorflow-addons tensorflow-serving-api


Documentation
=============

The documentation can be built using sphinx, or will be available at readthedocs.

Quick Start
===========

*Note*: In the next two sections, junn's call will just be abbreviated junn. Depending on whether Docker or a local Python installation should be used, this would mean either ``docker run --gpus all --ipc=host --tty --interactive --rm -v `pwd`:/data modsim/junn`` or ``python -m junn``.

Training
--------

For training a new neural network, first the network structure needs to be chosen, therefore the ``--NeuralNetwork`` parameter is used. Most commonly the ``Unet`` neural network will be chosen here, with various parameters being available defining the network structure (e.g. ``levels=4``, ``filters=64``, ``activation=relu`` to name a few). The model save path is passed via the ``--model`` parameter, which will contain a TensorFlow formatted model after training (which is itself a directory structure). Finally, an input of ground truth data is needed, for example an ImageJ TIFF file with ROIs denoting the desired structures, such as cells. Various tunable parameters can be set via the ``-t`` switch; a list of available tunables can be output by using the ``--tunables-show`` argument.

.. code-block:: bash

    > junn train --NeuralNetwork "Unet(optionA=1,optionB=2)" --model model_name -t SomeTunable=value training_data.tif

Upon call, the training will start, outputting the configured metrics at each epoch. If configured, outputs for TensorBoard will be written. Once the training is finished, or was interrupted by the user, e.g. because the achieved quality is good enough, the model is ready for prediction:

Prediction
----------

JUNN can take various input formats, such as Nikon ND2, Zeiss CZI, or OME-TIFF, and predict the image data, outputting either the raw probability masks from the neural networks, or detected objects as ImageJ ROIs.

.. code-block:: bash

    > junn predict --model model_name file_to_predict.tif --output result.tif --output-type roi

License
=======

JUNN is licensed under the 2-clause BSD License, see :doc:`LICENSE`.
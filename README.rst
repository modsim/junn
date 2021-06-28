JUNN Readme
===========

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

The simplest way to run JUNN is to use the Docker images provided.

Anaconda
--------

To install JUNN locally, we recommend using the Anaconda package manager.

Since Anaconda TensorFlow packages tend to lag a bit behind, it is necessary to install the official TensorFlow binaries from PyPI.

Github
------

Installing JUNN from Github is only recommended for users familiar with Python development.

Documentation
=============

See ... .

Quick Start
===========

Training
--------

Prediction
----------

License
=======

JUNN is licensed under the 2-clause BSD License, see :doc:`LICENSE`.
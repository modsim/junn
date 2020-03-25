Concepts
========

This page's aim is to convey some of JUNN's concepts, which might go amiss if the merely the API documentation is observed.

Network and training pipeline assembly by OOP & Mix-ins
-------------------------------------------------------

The core structure defining a network and training pipeline within JUNN is the NeuralNetwork class,
it has various member methods which can be overridden in order to quickly create a new type of network or altered training methodology.

To this extend, various steps, such as defining the network, defining the input augmentations, etc. are done by individual methods.

Furthermore, concepts, such as 'tile-based training' or 'augmentation' are done in individual mix-ins, which can just be inherited into a desired training class.

E.g. to implement a novel network following a tile-based prediction approach with the standard JUNN augmentation functionality, one can just subclass the NeuralNetwork class as follows:

TODO: Example

Strictly building a compute graph using TensorFlow primitives
-------------------------------------------------------------

Other tools often follow a mixed approach, where data is pre-processed within Python to specifically match a TensorFlow compute graph generated, which has the downside that (possibly slower) Python computation is used, and the resulting model remains highly dependent on the Python-based pre/postprocessing.

To avoid those potential downsides, JUNN tries to model all steps necessary for processing the data in pure TensorFlow: Once data is transferred into a .tfrecord file, processing is done solely by (performant C++/GPU-based) TensorFlow primitives: The data augmentation pipeline, as well as the training. As an additional gimmick, the prediction routines (such as tiling the image, performing normalization, etc.) are encoded via TensorFlow operations as well, yielding to a completely standalone usable model, which can readily be used in completely distinct tools, such as TensorFlow Serving or DeepImageJ.

To this extend, JUNN expects the programmer to define @tf.function decorated functions for various purposes, which can be completely transformed into a TensorFlow compute graph.

TODO: Example

Furthermore, the prediction part of JUNN makes use of these properties: Instead of completely reinstantiating the model, only the graph is loaded in lightweight manner and execution is left to TensorFlow. By this unified approach, the various prediction backends (direct, via GRPC/HTTP and TensorFlow Serving) were easily implementable.

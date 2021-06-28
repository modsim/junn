"""The train module."""
from tunable import Tunable


class Epochs(Tunable):
    """Epochs to train the network."""

    default = 200


class StepsPerEpoch(Tunable):
    """Steps to perform per epoch (should be roughly len(datasets)/len(batchsize))."""

    default = 300


class BatchSize(Tunable):
    """Batch size."""

    default = 5


class ValidationBatchSize(Tunable):
    """Validation batch size."""

    default = 5


class ValidationSteps(Tunable):
    """Steps of validation."""

    default = 50


class Optimizer(Tunable):
    """Optimizer."""

    default = "Adam"

    @classmethod
    def test(cls, value):  # noqa: D102,ANN102,ANN001,ANN206
        return value.lower() in {'adam', 'rmsprop', 'sgd'}


class LearningRate(Tunable):
    """Learning rate."""

    # default = 0.001
    default = 1.0e-5


class Momentum(Tunable):
    """SGD optimizer momentum."""

    default = 0.0


class Decay(Tunable):
    """SGD optimizer decay."""

    default = 0.0


class Metrics(Tunable):
    """Metrics to use."""

    default = "f_score,dice_index,jaccard_index"

    @classmethod
    def test(cls, value):  # noqa: D102,ANN102,ANN001,ANN206
        try:
            cls.get_list(value)
        except AttributeError:
            return False
        return True

    @classmethod
    def get_list(cls, values=None):  # noqa: D102,ANN102,ANN001,ANN206
        if values is None:  # prevent recursion by using passed value in test case
            values = cls.value
        from ..common import losses

        values = [value.strip() for value in values.split(',')]
        return [getattr(losses, value) for value in values]


class DatasetGenerationBenchmarkCount(Tunable):
    """How many samples should be generated for dataset benchmarking."""

    default = 1000


class Profile(Tunable):
    """See documentation of the profile_batch parameter of the TensorBoard callback."""

    default = 0


class PreprocessingMapParallel(Tunable):
    """Whether preprocessing should be performed parallel."""

    default = False

    @classmethod
    def prepared(cls):  # noqa: D102,ANN102,ANN001,ANN206
        import tensorflow as tf

        if cls.value is False:
            return None
        elif cls.value is True:
            return tf.data.experimental.AUTOTUNE
        else:
            return cls.value


# output helpers


class TensorBoardHistogramFrequency(Tunable):
    """TensorBoard callback histogram output frequency."""

    default = 0


class TensorBoardWriteGradients(Tunable):
    """Whether to write gradients with the TensorBoard callback."""

    default = False


class WriteLearningCSV(Tunable):
    """If set, write to this file CSV log containing learning information."""

    default = ""


class TensorBoardSegmentationDataset(Tunable):
    """Test segment this TIFF file and add the output to TensorBoard."""

    default = ""


class TensorBoardSegmentationEpochs(Tunable):
    """Test segment every n epochs and add the output to TensorBoard."""

    default = 10

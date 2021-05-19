from tunable import Tunable


class Epochs(Tunable):
    """ Epochs to train the network """

    default = 200


class StepsPerEpoch(Tunable):
    """ Steps to perform per epoch (should be roughly len(datasets)/len(batchsize)) """

    default = 300


class BatchSize(Tunable):
    """ Batch size """

    default = 5


class ValidationBatchSize(Tunable):
    """ Validation batch size """

    default = 5


class ValidationSteps(Tunable):
    """ Steps of validation """

    default = 50


class Optimizer(Tunable):
    """ Optimizer """

    default = "Adam"

    @classmethod
    def test(cls, value):
        return value.lower() in {'adam', 'rmsprop', 'sgd'}


class LearningRate(Tunable):
    """ Learning rate """

    # default = 0.001
    default = 1.0e-5


class Momentum(Tunable):
    """ SGD optimizer momentum """

    default = 0.0


class Decay(Tunable):
    """ SGD optimizer decay """

    default = 0.0


class Metrics(Tunable):
    default = "f_score,dice_index,jaccard_index"

    @classmethod
    def test(cls, value):
        try:
            cls.get_list(value)
        except AttributeError:
            return False
        return True

    @classmethod
    def get_list(cls, values=None):
        if values is None:  # prevent recursion by using passed value in test case
            values = cls.value
        from ..common import losses

        values = [value.strip() for value in values.split(',')]
        return [getattr(losses, value) for value in values]


class DatasetGenerationBenchmarkCount(Tunable):
    default = 1000


class Profile(Tunable):
    default = 0


class PreprocessingMapParallel(Tunable):
    default = False

    @classmethod
    def prepared(cls):
        import tensorflow as tf

        if cls.value is False:
            return None
        elif cls.value is True:
            return tf.data.experimental.AUTOTUNE
        else:
            return cls.value


# output helpers


class TensorBoardHistogramFrequency(Tunable):
    """  """

    default = 0


class TensorBoardWriteGradients(Tunable):
    """  """

    default = False


class WriteLearningCSV(Tunable):
    """ If set, write to this file CSV log containing learning information """

    default = ""


class TensorBoardSegmentationDataset(Tunable):
    """ Perform a test segmentation with this TIFF file and add the output to TensorBoard """

    default = ""


class TensorBoardSegmentationEpochs(Tunable):
    """ Perform a test segmentation every n epochs and add the output to TensorBoard """

    default = 10

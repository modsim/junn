import numpy as np
import pytest

from junn.networks import NeuralNetwork, tf_function_nop
from junn.networks.all import *


def network_test(net: NeuralNetwork):
    net.setup_model()


@pytest.mark.parametrize(
    'network,parameters,raises',
    [
        (Unet, '', None),
        (Unet, 'tile_size=96', None),
        (Unet, 'categorical=True', None),
        (Unet, 'recurrent_num=2', None),
        (Unet, 'residual_connections=True', None),
        (Unet, 'just_convolutions=True', None),
        (Unet, 'no_transpose=True', None),
        (Unet, 'dropout_down=0.5', None),
        (LinkNet, '', None),
        (LinkNet, 'batch_normalization=True', None),
        (LinkNet, 'categorical=True', NotImplementedError),
        (MSD, '', None),
        (MSD, 'categorical=True', None),
    ],
)
def test_network(network, parameters, raises):
    parameters = 'dict(' + parameters + ')'
    parameters = eval(parameters)

    if raises:
        with pytest.raises(raises):
            network_test(network(**parameters))
    else:
        network_test(network(**parameters))


def test_raise_codecoverage_nn():
    n = object.__new__(NeuralNetwork)

    n.init()

    with pytest.raises(RuntimeError):
        n.get_model()

    n.model = tf_function_nop

    n.get_training_fn()

    n.get_raw_fn()

    n.get_signatures()

    n.prediction_fn = n.get_prediction_fn()

    assert (n.predict(np.zeros((256, 256))) == np.zeros((256, 256))).all()

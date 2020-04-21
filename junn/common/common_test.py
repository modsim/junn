import pytest

def test_distributed():
    import junn.common.distributed


    def _inner():
        from junn.common.distributed import (
            init, local_rank, rank, is_running_distributed, size, is_rank_zero, barrier)

        init()
        assert local_rank() == 0
        assert rank() == 0
        assert is_rank_zero()
        assert size() == 1
        assert is_running_distributed() == False
        barrier('testing')
        assert True

    old_hvd = junn.common.distributed.hvd

    _inner()

    junn.common.distributed.hvd = None

    _inner()

    junn.common.distributed.hvd = old_hvd


def test_faked_mpirun():
    import os
    os.environ['OMPI_COMM_WORLD_SIZE'] = '1'

    from junn.common.distributed import init
    init()


def test_missing_horovod():
    import sys
    sys.modules['horovod.tensorflow.keras'] = None
    if 'junn.common.distributed' in sys.modules:
        del sys.modules['junn.common.distributed']
        import junn.common.distributed


def test_horovod_wo_gpu(disable_cuda):
    with disable_cuda():
        # TODO: this only runs properly if TF has not been imported yet
        import os
        print(os.environ)
        import sys
        print(sys.modules.keys())
        import tensorflow as tf
        print(tf.config.get_visible_devices('GPU'))
        import junn.common.distributed
        junn.common.distributed.init()


def test_layer_runmodeltiled():
    from junn.common.layers.run_model_layer import RunModelTiled
    import numpy as np

    fake_model = lambda a: a + 10

    image = np.random.random((1, 512, 512, 1))

    rmt = RunModelTiled(model=fake_model)

    result = rmt.call(image)

    assert (result == (image + 10)).numpy().all()


def test_layer_runmodeltiled_overlap():
    from junn.common.layers.run_model_layer import RunModelTiled
    import numpy as np

    fake_model = lambda a: a + 10

    image = np.random.random((1, 512, 512, 1))

    rmt = RunModelTiled(model=fake_model, overlap=(16, 16))

    result = rmt.call(image)

    assert (result == (image + 10)).numpy().all()


def test_layer_runmodeltiled_nomodel():
    from junn.common.layers.run_model_layer import RunModelTiled
    import numpy as np

    image = np.random.random((1, 512, 512, 1))

    rmt = RunModelTiled(model=None)

    result = rmt.call(image)

    assert (result == image).numpy().all()


@pytest.mark.parametrize('loss,a,b,expected', [
    ('dice_loss', 0.0, 0.0, 0.0),
    ('dice_loss', 0.0, 1.0, 0.0),
    ('dice_loss', 1.0, 0.0, 0.0),
    ('dice_loss', 1.0, 1.0, -1.0),

    ('dice_index_direct', 0.0, 0.0, 0.0),
    ('dice_index_direct', 0.0, 1.0, 0.0),
    ('dice_index_direct', 1.0, 0.0, 0.0),
    ('dice_index_direct', 1.0, 1.0, 1.0),

    ('accuracy', 0.0, 0.0, 1.0),
    ('accuracy', 0.0, 1.0, 0.0),
    ('accuracy', 1.0, 0.0, 0.0),
    ('accuracy', 1.0, 1.0, 1.0),

    ('precision', 0.0, 0.0, 0.0),
    ('precision', 0.0, 1.0, 0.0),
    ('precision', 1.0, 0.0, 0.0),
    ('precision', 1.0, 1.0, 1.0),

    ('recall', 0.0, 0.0, 0.0),
    ('recall', 0.0, 1.0, 0.0),
    ('recall', 1.0, 0.0, 0.0),
    ('recall', 1.0, 1.0, 1.0),

    ('f_score', 0.0, 0.0, 0.0),
    ('f_score', 0.0, 1.0, 0.0),
    ('f_score', 1.0, 0.0, 0.0),
    ('f_score', 1.0, 1.0, 1.0),

    ('dice_index', 0.0, 0.0, 0.0),
    ('dice_index', 0.0, 1.0, 0.0),
    ('dice_index', 1.0, 0.0, 0.0),
    ('dice_index', 1.0, 1.0, 1.0),

    ('tversky_index', 0.0, 0.0, 0.0),
    ('tversky_index', 0.0, 1.0, 0.0),
    ('tversky_index', 1.0, 0.0, 0.0),
    ('tversky_index', 1.0, 1.0, 1.0),

])
def test_losses(loss, a, b, expected):
    import numpy as np
    import junn.common.losses
    loss_func = getattr(junn.common.losses, loss)

    size = 128

    buffer = np.zeros((size, size), dtype=np.float32)

    a_buffer = buffer.copy()
    a_buffer.fill(a)

    b_buffer = buffer.copy()
    b_buffer.fill(b)

    result = loss_func(a_buffer, b_buffer)

    if expected == 'nan':
        assert result != result
    else:
        assert result == expected

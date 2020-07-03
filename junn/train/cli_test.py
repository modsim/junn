#
# with disable_cuda():
#     import tensorflow as tf

from junn.common.configure_tensorflow import configure_tensorflow
configure_tensorflow(0)


import tensorflow as tf

# tf.config.experimental_run_functions_eagerly(True)

import pytest

from junn.train.cli import main


def test_main_no_args():
    with pytest.raises(SystemExit):
        main([])


def test_main_dataset_benchmark(tmpdir, empty_training_data):
    main(['--NeuralNetwork', 'Unet',
          '--output-dataset', str(tmpdir.join('benchmark.tif')),
          '--output-dataset-count', '32',
          empty_training_data
          ])


def check_filenames(model_path):
    existing_filenames = set([filename.basename for filename in model_path.listdir()])
    expected_filenames = {'saved_model.pb', 'checkpoint', 'cp.ckpt.index', 'train',
                          'resultImage.tiff', 'exampleImage.tiff', 'postprocessing.txt',
                          'preprocessing.txt', 'config.xml', 'assets', 'variables'}

    expected_filenames |= {'training_data'}

    maybe_filenames = {'cp.ckpt.data-00000-of-00001', 'cp.ckpt.data-00000-of-00002', 'cp.ckpt.data-00001-of-00002'}

    assert (existing_filenames - expected_filenames - maybe_filenames) == set()


def test_main_large_integration(tmpdir, empty_training_data):
    neural_network = 'Unet(levels=2)'

    real_model_path = tmpdir.join('model_'+neural_network+'_test')
    model_path = tmpdir.join('model_%NeuralNetwork_%tag')

    args = [
        '--NeuralNetwork', neural_network, '--tag', 'test', '--model', str(model_path), empty_training_data,
        '--embed',
        '-t', 'Epochs=1',
        '-t', 'StepsPerEpoch=4',
        '-t', 'BatchSize=1',
        '-t', 'TensorBoardSegmentationDataset=' + empty_training_data,
        '-t', 'TensorBoardSegmentationEpochs=1']

    main(args)

    check_filenames(real_model_path)

    main(args + ['--resume'])


# noinspection DuplicatedCode
def test_main_image_directory(tmpdir, empty_training_image_directory):
    neural_network = 'Unet(levels=2)'

    model_path = tmpdir.join('model')

    images, labels = (
                         empty_training_image_directory + '/images/image-{}.png',
                         empty_training_image_directory + '/labels/label-{}.png'
    )

    args = [
        '--NeuralNetwork', neural_network, '--model', str(model_path), empty_training_image_directory,
        '-t', 'Epochs=1',
        '-t', 'StepsPerEpoch=4',
        '--TrainingInput', 'ImageDirectoryInput(images=%s,labels=%s)' % (images, labels)]

    main(args)

    check_filenames(model_path)

    main(args + ['--resume'])


# noinspection DuplicatedCode
def test_main_image_directory_hvd_disable(tmpdir, empty_training_image_directory):
    neural_network = 'Unet(levels=2)'

    model_path = tmpdir.join('model')

    images, labels = (
                         empty_training_image_directory + '/images/image-{}.png',
                         empty_training_image_directory + '/labels/label-{}.png'
    )

    args = [
        '--NeuralNetwork', neural_network, '--model', str(model_path), empty_training_image_directory,
        '-t', 'Epochs=1',
        '-t', 'StepsPerEpoch=4',
        '--TrainingInput', 'ImageDirectoryInput(images=%s,labels=%s)' % (images, labels)]

    import junn.common.distributed

    old_hvd = junn.common.distributed.hvd

    junn.common.distributed.hvd = None

    main(args)

    check_filenames(model_path)

    main(args + ['--resume'])

    junn.common.distributed.hvd = old_hvd

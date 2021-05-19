import warnings


def try_load_tensorflow_addons():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            import tensorflow_addons.activations as tfa_activations
            from tensorflow.keras.utils import get_custom_objects

            # bring TensorFlow Addons activations into 'normal' namespace
            for activation in dir(tfa_activations):
                decorated = 'Addons>%s' % (activation,)
                if decorated in get_custom_objects():
                    get_custom_objects()[activation] = get_custom_objects()[decorated]

            from tensorflow_addons.register import register_all

            register_all()
    except ImportError:
        pass

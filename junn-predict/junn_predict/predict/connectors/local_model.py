from ...common.configure_tensorflow import configure_tensorflow
from ...common.tensorflow_addons import try_load_tensorflow_addons
from .model_connector import ModelConnector


class LocalModel(ModelConnector, ModelConnector.Default):
    @staticmethod
    def check_import():
        pass

    def __init__(self, arg):
        super().__init__(arg)

        configure_tensorflow(seed=0)

        import tensorflow as tf

        try_load_tensorflow_addons()

        if arg.startswith('http://') or arg.startswith('https://'):
            import tensorflow_hub as hub

            self.model = hub.load(arg)
        else:
            self.model = tf.saved_model.load(arg)

    def get_signatures(self):
        return list(sorted(self.model.signatures.keys()))

    def call(self, signature, data):
        import tensorflow as tf

        data = tf.convert_to_tensor(data)

        result = self.model.signatures[signature](data)

        if len(result) == 1:
            return next(iter(result.values()))

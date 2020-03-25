from .model_connector import ModelConnector


class LocalModel(ModelConnector, ModelConnector.Default):
    @staticmethod
    def check_import():
        pass

    def __init__(self, arg):
        super().__init__(arg)
        import tensorflow as tf
        self.model = tf.saved_model.load(arg)

    def get_signatures(self):
        return list(sorted(self.model.signatures.keys()))

    def call(self, signature, data):
        import tensorflow as tf

        data = tf.convert_to_tensor(data)

        result = self.model.signatures[signature](data)

        if len(result) == 1:
            return next(iter(result.values()))

import tensorflow as tf


class PerImageStandardizationPreprocessingMixin:
    # noinspection PyMethodMayBeStatic
    def get_raw_fn(self):
        import tensorflow.python.util.deprecation as deprecation

        @tf.function
        def preprocess(image):
            image = tf.cast(image, tf.float32)
            # TF 2.1 and it is still not in release
            with deprecation.silence():  # can be removed once TF PR # 34807 is merged
                image = tf.image.per_image_standardization(image)
            return image

        return preprocess

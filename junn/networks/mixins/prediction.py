import tensorflow as tf


class PredictionSignatureMixin:
    def get_signatures(self):
        predict = self.prediction_fn

        signatures = {}

        @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)])
        def predict_no_channel(image):
            return predict(image[..., tf.newaxis])[:, :, 0]

        signatures['predict_no_channel'] = predict_no_channel

        @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.string)])
        def predict_png(png_data):
            image = tf.io.decode_png(png_data)

            image = tf.cast(image, tf.float32)
            prediction = predict(image)

            # either pre multiply, or post multiply
            # one yields a gray scale image, the other basically only a binary b/w image

            prediction = prediction * 255
            prediction = tf.cast(prediction, tf.uint8)

            prediction = tf.image.encode_png(prediction, compression=0)

            return dict(predict_png_bytes=prediction)  # important _bytes suffix for TF Serving

        signatures['predict_png'] = predict_png

        return signatures

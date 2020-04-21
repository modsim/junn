import tensorflow as tf

from junn.common.layers.run_model_layer import RunModelTiled


class TilebasedPredictionMixin:
    def get_prediction_fn(self):
        model = self.model
        tile_size = self.tile_size

        prediction_tile_size = tile_size[0:2]
        overlap = (32, 32)  # None  # (32, 32)

        @tf.function(input_signature=[tf.TensorSpec([None, None, None], dtype=tf.float32)])
        def _predict(image):
            image = self.get_raw_fn()(image)
            image = tf.expand_dims(image, axis=0)

            # image = tf.cond(
            #     tf.shape(image)[3] > 1,
            #     lambda: tf.expand_dims(tf.reduce_mean(image, axis=3), axis=-1),
            #     lambda: image
            # )

            if tf.shape(image)[3] > 1:
                image = tf.reduce_mean(image, axis=3)
                image = tf.expand_dims(image, axis=-1)

            prediction = RunModelTiled(
                model=model, block_size=prediction_tile_size, batch_size=32, overlap=overlap
            )(image)

            prediction = prediction[0]

            return tf.identity(prediction, name='prediction')

        return _predict

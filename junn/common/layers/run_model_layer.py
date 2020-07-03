import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class RunModelTiled(Layer):
    """
    Runs a model with a fixed input size in a tiled manner over the (larger) input tensor (image).
    """
    def __init__(self,
                 model=None,
                 block_size=(128, 128,),
                 overlap=(0, 0,),
                 batch_size=16,
                 parallel_iterations=1,
                 dtype=tf.float32,
                 # overlap_ontop=False
                 ):
        """
        Instantiate the layer.
        :param model: Keras model
        :param block_size: block size (n, m)
        :param overlap: overlap (n, m)
        :param batch_size: Inner batch size
        :param parallel_iterations: Whether parallel iterations should be used
        :param dtype: Output dtype
        """
        if model is None:
            self.model = lambda input_: input_
        else:
            self.model = model

        self.block_size = block_size

        self.target_dtype = dtype
        # self.output_size = block_size

        if overlap == (0, 0) or overlap is None:
            self.overlap = None
        else:
            self.overlap = overlap

        self.batch_size = batch_size
        self.parallel_iterations = parallel_iterations

        super(RunModelTiled, self).__init__()

    @staticmethod
    def _pad_to(tensor, first_dim):
        """
        Helper function to pad a tensor.
        :param tensor:
        :param first_dim:
        :return:
        """
        tensor_shape = tf.shape(tensor)
        paddings = tf.concat([
            tf.convert_to_tensor([[0, tf.cast(first_dim - tensor_shape[0], tf.int32)]]),
            tf.zeros([tf.shape(tensor_shape)[0] - 1, 2], dtype=tf.int32)
        ], axis=0)

        return tf.pad(tensor, paddings)

    def call(self, raw_input_tensor, **kwargs):
        """
        Inner function called by Keras.
        :param raw_input_tensor:
        :param kwargs:
        :return:
        """
        # set variables, original input
        raw_input_tensor_shape = tf.shape(raw_input_tensor)

        # calculate necessary crops to have a padded result
        
        crops = [[0, self.block_size[0] - (raw_input_tensor_shape[1] % self.block_size[0])],
                 [0, self.block_size[1] - (raw_input_tensor_shape[2] % self.block_size[1])]]

        # first generation of batch
        if self.overlap is None:
            raw_batched_input = tf.space_to_batch_nd(raw_input_tensor, self.block_size, crops)
        else:
            raw_batched_input = tf.image.extract_patches(
                raw_input_tensor,
                sizes=(1, self.block_size[0], self.block_size[1], 1,),
                strides=(1, self.block_size[0] - self.overlap[0], self.block_size[1] - self.overlap[1], 1,),
                rates=(1, 1, 1, 1),
                padding='SAME')
            raw_batched_input = tf.transpose(raw_batched_input, [3, 1, 2, 0])

        raw_batched_input_shape = tf.shape(raw_batched_input)

        # reorder dimensions 
        intermediate_batched_input = tf.transpose(raw_batched_input, [1, 2, 0, 3])
        intermediate_batched_input_shape = tf.shape(intermediate_batched_input)

        batched_input_shape = (
            (intermediate_batched_input_shape[0] * intermediate_batched_input_shape[1],) +
            self.block_size +
            (raw_input_tensor_shape[-1],)
        )

        batched_input = tf.reshape(intermediate_batched_input, batched_input_shape)

        # calculate the batch actual batch steps
        target_batch_count = (
                (batched_input_shape[0] // self.batch_size) +
                tf.cond(
                    tf.equal((batched_input_shape[0] % self.batch_size), 0),
                    lambda: 0,
                    lambda: 1
                )
        )

        collector_tensor = tf.TensorArray(self.target_dtype, size=target_batch_count)

        _, collector_tensor = tf.while_loop(
            cond=lambda i, _: i < target_batch_count,
            body=lambda i, ct: (i + 1,
                                ct.write(i,
                                         tf.cast(self._pad_to(self.model(
                                             batched_input[
                                                 i * self.batch_size:
                                                 tf.minimum(
                                                     i * self.batch_size + self.batch_size,
                                                     batched_input_shape[0]
                                                 )
                                             ]), self.batch_size), self.target_dtype)
                                         )
                                ),
            loop_vars=(0, collector_tensor),
            parallel_iterations=self.parallel_iterations,
            swap_memory=False,
        )

        result = collector_tensor.stack()

        # reconsider this shape assembly
        new_last_dim = tf.shape(result)[-1]

        raw_batched_input_shape = tf.concat([raw_batched_input_shape[:-1], (new_last_dim,)], axis=0)
        batched_input_shape = tf.concat([batched_input_shape[:-1], (new_last_dim,)], axis=0)

        new_result_shape = tf.shape(result)

        if self.overlap:
            result = result[:, :, self.overlap[0]//2:-self.overlap[0]//2, self.overlap[1]//2:-self.overlap[0]//2, :]

            new_result_shape = tf.shape(result)

            batched_input_shape = (
                    (intermediate_batched_input_shape[0] * intermediate_batched_input_shape[1],) +
                    (self.block_size[0] - self.overlap[0], self.block_size[1] - self.overlap[1]) +
                    (new_last_dim,)
            )
            raw_batched_input_shape = tf.concat([
                [(self.block_size[0] - self.overlap[0]) * (self.block_size[1] - self.overlap[1])],
                raw_batched_input_shape[1:],
            ], axis=0)

        new_shape = tf.concat([
            tf.convert_to_tensor([new_result_shape[0] * new_result_shape[1], ])[..., tf.newaxis],
            new_result_shape[2:][..., tf.newaxis]
        ], axis=0)

        new_shape = new_shape[:, 0]

        result = tf.reshape(result, new_shape)

        result = tf.slice(result, tf.zeros_like(batched_input_shape), batched_input_shape)

        result = tf.transpose(result, [1, 2, 0, 3])

        reshaped_result = tf.reshape(result, raw_batched_input_shape)
        if self.overlap is None:
            reassembled_result = tf.batch_to_space(reshaped_result, self.block_size, crops=crops)
        else:
            # TODO: this does not 100% make sense, since apparently at some point W/H are swapped in some of the tf ops
            # crop_dim_1 = ((raw_input_tensor_shape[1] % (self.block_size[0] - self.overlap[0])) - self.overlap[0]) // 2
            # crop_dim_2 = ((raw_input_tensor_shape[2] % (self.block_size[1] - self.overlap[1])) - self.overlap[1]) // 2
            # TODO: surprise: it's not working...
            # TODO: or does it? 2020-01-06
            # <--
            # tf.print("\n", crop_dim_1, crop_dim_2, raw_input_tensor_shape)

            # crops = [[crop_dim_2, crop_dim_2], [crop_dim_1, crop_dim_1]]
            # crops = [[crop_dim_1, crop_dim_1], [crop_dim_2, crop_dim_2]]

            crops = [[0, 0], [0, 0]]

            reassembled_result = tf.batch_to_space(
                reshaped_result,
                (self.block_size[0] - self.overlap[0], self.block_size[1] - self.overlap[1]),
                crops=crops)

            reassembled_result_shape = tf.shape(reassembled_result)

            # tf.print((raw_input_tensor_shape[1] - reassembled_result_shape[1])//2)
            # tf.print(-(raw_input_tensor_shape[1] - reassembled_result_shape[1])//2)

            reassembled_result = reassembled_result[
                :,
                (reassembled_result_shape[1] - raw_input_tensor_shape[1])//2:
                -(reassembled_result_shape[1] - raw_input_tensor_shape[1])//2,
                (reassembled_result_shape[2] - raw_input_tensor_shape[2])//2:
                -(reassembled_result_shape[2] - raw_input_tensor_shape[2])//2,
                :
            ]

            # <--
            # tf.print(tf.shape(reassembled_result))

        return reassembled_result

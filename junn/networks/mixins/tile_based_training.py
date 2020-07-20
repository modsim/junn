import tensorflow as tf
import tensorflow_addons as tfa

from ...common.functions import get_gaussian_kernels, pad_to, convolve
from ...common.functions.affine import tfm_identity, tfm_reflect, tfm_rotate, tfm_shear, tfm_scale, \
    tfm_to_tf_transform
from .augmentation import BlurMinimumSigma, BlurMaximumSigma, BlurSteps, BlurProbability, ReflectProbability, \
    ReflectXProbability, ReflectYProbability, RotateProbability, RotateMinimum, RotateMaximum, ShearProbability, \
    ShearXAngleMinimum, ShearXAngleMaximum, ShearYAngleMinimum, ShearYAngleMaximum, ScaleProbability, ScaleMinimum, \
    ScaleMaximum


class TilebasedTrainingMixin:
    def get_training_fn(self, validation: bool = False):
        tile_size = self.tile_size
        factor = 2
        big_block = (factor * tile_size[0], factor * tile_size[1], tile_size[2],)

        margins = (big_block[0] - tile_size[0], big_block[1] - tile_size[1])

        gaussian_kernels = get_gaussian_kernels(BlurMinimumSigma.value, BlurMaximumSigma.value, BlurSteps.value)

        @tf.function
        def create_training_data(image, labels):

            image, labels = tf.cast(image, tf.float32), tf.cast(labels, tf.float32)

            # resize image and labels to larger (ie. big_block) size, so we have enough data to work with

            # TODO: This fails if the input data is smaller than the tile size! (or smaller than twice the tile size?)

            image, labels = pad_to(image, big_block), pad_to(labels, big_block)

            # randomly crop out a big_block

            image = tf.image.random_crop(image, size=big_block, seed=10)
            labels = tf.image.random_crop(labels, size=big_block, seed=10)

            # helper function, because we'll need a lot of random scalar tf.float32
            def random_float(minval=0.0, maxval=1.0, seed=None):
                return tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.float32, seed=seed)

            # blur
            # this could be done with tfa.image.gaussian_filter2d, but in first tests, it was slower
            if random_float(seed=20) < BlurProbability.value:
                gaussian_kernel_n = tf.random.uniform(
                    shape=[], minval=0, maxval=len(gaussian_kernels), dtype=tf.int32, seed=21)
                gaussian_kernel = tf.gather(gaussian_kernels, gaussian_kernel_n)
                convolve(tf.cast(image, tf.float32), kernel=gaussian_kernel)

            # construct a transformation matrix, starting with an identity matrix ...

            current_shape = tf.shape(image)
            mat = tfm_identity()

            # reflection

            if random_float(seed=30) < ReflectProbability.value:
                mat @= tfm_reflect(
                    1.0 if random_float(seed=31) < ReflectXProbability.value else 0.0,
                    1.0 if random_float(seed=32) < ReflectYProbability.value else 0.0,
                    shape=current_shape)

            # rotation

            if random_float(seed=40) < RotateProbability.value:
                mat @= tfm_rotate(
                    random_float(minval=RotateMinimum.value, maxval=RotateMaximum.value, seed=41),
                    shape=current_shape)

            # shearing

            if random_float(seed=50) < ShearProbability.value:
                mat @= tfm_shear(
                    random_float(minval=ShearXAngleMinimum.value, maxval=ShearXAngleMaximum.value, seed=51),
                    random_float(minval=ShearYAngleMinimum.value, maxval=ShearYAngleMaximum.value, seed=52),
                    shape=current_shape)

            # scaling

            if random_float(seed=60) < ScaleProbability.value:
                mat @= tfm_scale(
                    random_float(minval=ScaleMinimum.value, maxval=ScaleMaximum.value, seed=61),
                    shape=current_shape)

            # convert and apply the matrix

            theta_8 = tfm_to_tf_transform(mat)

            image = tfa.image.transform(image, theta_8, interpolation='BILINEAR')
            labels = tfa.image.transform(labels, theta_8, interpolation='NEAREST')

            image = image[margins[0]//2:-margins[0]//2, margins[1]//2:-margins[1]//2, :]
            # tf.image.random_crop(image, size=tile_size, seed=3)
            labels = labels[margins[0]//2:-margins[0]//2, margins[1]//2:-margins[1]//2, :]
            # tf.image.random_crop(labels, size=tile_size, seed=3)

            # should we cast here? it seems to make the next steps less flexible
            # image, labels = tf.cast(image, tf.float16), tf.cast(labels, tf.float16)

            return image, labels

        return create_training_data

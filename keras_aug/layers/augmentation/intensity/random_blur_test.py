import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers


class RandomBlurTest(tf.test.TestCase, parameterized.TestCase):
    regular_args = {
        "factor": (3, 7),
    }
    no_aug_args = {
        "factor": (1, 1),
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = layers.RandomBlur(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.RandomBlur(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = layers.RandomBlur(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

import tensorflow as tf
from absl.testing import parameterized

from keras_aug.datapoints import image


class ImageOpsTest(tf.test.TestCase, parameterized.TestCase):
    def test_blend(self):
        ones = tf.ones(shape=(2, 4, 4, 3))
        twos = tf.ones(shape=(2, 4, 4, 3)) * 2
        ratios = tf.ones(shape=(2, 1, 1, 1)) * 0.3
        expected_result = ratios * ones + (1.0 - ratios) * twos

        result = image.blend(twos, ones, ratios)

        self.assertAllEqual(result, expected_result)

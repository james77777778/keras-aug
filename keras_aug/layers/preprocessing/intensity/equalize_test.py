import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_aug import layers


class EqualizeTest(tf.test.TestCase, parameterized.TestCase):
    def test_return_shapes_inside_model(self):
        layer = layers.Equalize(value_range=(0, 255))
        inp = keras.layers.Input(shape=[4, 4, 5])
        out = layer(inp)
        model = keras.models.Model(inp, out)

        self.assertEqual(model.layers[-1].output_shape, (None, 4, 4, 5))

    def test_equalizes_to_all_bins(self):
        xs = tf.random.uniform((2, 32, 32, 3), 0, 255, seed=2023)
        layer = layers.Equalize(value_range=(0, 255))
        xs = layer(xs)

        for i in range(0, 256):
            self.assertTrue(tf.math.reduce_any(xs == i))

    @parameterized.named_parameters(
        ("float32", tf.float32), ("int32", tf.int32), ("int64", tf.int64)
    )
    def test_input_dtypes(self, dtype):
        xs = tf.random.uniform((2, 32, 32, 3), 0, 255, dtype=dtype, seed=2023)
        layer = layers.Equalize(value_range=(0, 255))
        xs = layer(xs)

        for i in range(0, 256):
            self.assertTrue(tf.math.reduce_any(xs == i))
        self.assertAllInRange(xs, 0, 255)

    @parameterized.named_parameters(("0_255", 0, 255), ("0_1", 0, 1))
    def test_output_range(self, lower, upper):
        xs = tf.random.uniform((2, 4, 4, 3), lower, upper, dtype=tf.float32)
        layer = layers.Equalize(value_range=(lower, upper))
        xs = layer(xs)
        self.assertAllInRange(xs, lower, upper)

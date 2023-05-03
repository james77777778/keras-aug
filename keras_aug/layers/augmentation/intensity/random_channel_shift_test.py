import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers


class RandomChannelShiftTest(tf.test.TestCase, parameterized.TestCase):
    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 8, 3)), tf.ones((4, 8, 3))],
                axis=0,
            ),
            dtype=tf.float32,
        )
        layer = layers.RandomChannelShift(
            factor=[0.1, 0.3], value_range=(0, 255)
        )

        xs = layer(xs, training=True)
        self.assertFalse(tf.math.reduce_any(xs[0] == 2.0))
        self.assertFalse(tf.math.reduce_any(xs[1] == 1.0))

    def test_5_channels(self):
        xs = tf.cast(tf.ones((4, 4, 5)), dtype=tf.float32)
        layer = layers.RandomChannelShift(
            factor=0.4, channels=5, value_range=(0, 255)
        )
        xs = layer(xs, training=True)
        self.assertFalse(tf.math.reduce_any(xs == 1.0))

    def test_1_channel(self):
        xs = tf.cast(tf.ones((4, 4, 1)), dtype=tf.float32)
        layer = layers.RandomChannelShift(
            factor=0.4, channels=1, value_range=(0, 255)
        )
        xs = layer(xs, training=True)
        self.assertFalse(tf.math.reduce_any(xs == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(tf.ones((4, 4, 3)), dtype=tf.float32)
        layer = layers.RandomChannelShift(factor=0.4, value_range=(0, 255))
        xs = layer(xs, training=True)
        self.assertFalse(tf.math.reduce_any(xs == 1.0))

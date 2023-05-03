import tensorflow as tf

from keras_aug import layers


class ChannelShuffleTest(tf.test.TestCase):
    def test_channel_shuffle_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((40, 40, 1)), 2 * tf.ones((40, 40, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = layers.ChannelShuffle(groups=1)
        xs = layer(xs)
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_channel_shuffle_call_results_multi_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((40, 40, 20)), 2 * tf.ones((40, 40, 20))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = layers.ChannelShuffle(groups=5)
        xs = layer(xs)
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((128, 64, 1)), tf.ones((128, 64, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = layers.ChannelShuffle(groups=1)
        xs = layer(xs)
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(tf.ones((128, 128, 1)), dtype=tf.float32)

        layer = layers.ChannelShuffle(groups=1)
        xs = layer(xs)
        self.assertTrue(tf.math.reduce_any(xs == 1.0))

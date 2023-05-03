import tensorflow as tf

from keras_aug import layers


class RandomGridMaskTest(tf.test.TestCase):
    def test_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((80, 80, 1)), 2 * tf.ones((80, 80, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )
        fill_value = 0.0
        layer = layers.RandomGridMask(
            size_factor=(0.7, 1.0),
            ratio_factor=0.3,
            rotation_factor=(20, 30),
            fill_mode="constant",
            fill_value=fill_value,
            seed=2023,
        )

        xs = layer(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((80, 40, 1)), tf.ones((80, 40, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )
        fill_value = 100.0
        layer = layers.RandomGridMask(
            size_factor=(0.5, 1.0),
            ratio_factor=0.6,
            rotation_factor=30,
            fill_mode="constant",
            fill_value=fill_value,
            seed=2023,
        )

        xs = layer(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((8, 8, 1)),
            dtype=tf.float32,
        )
        layer = layers.RandomGridMask(
            ratio_factor=(0.5, 0.5), fill_mode="constant", fill_value=0.0
        )

        xs = layer(xs)

        self.assertTrue(tf.math.reduce_any(xs == 0.0))
        self.assertTrue(tf.math.reduce_any(xs == 1.0))

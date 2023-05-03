import tensorflow as tf

from keras_aug import layers


class GrayscaleTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 4, 8, 3))

        layer = layers.Grayscale(output_channels=1)
        xs1 = layer(xs)

        layer = layers.Grayscale(output_channels=3)
        xs2 = layer(xs)

        self.assertEqual(xs1.shape, [2, 4, 8, 1])
        self.assertEqual(xs2.shape, [2, 4, 8, 3])

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((4, 8, 3)), tf.ones((4, 8, 3))], axis=0),
            tf.float32,
        )

        layer = layers.Grayscale(output_channels=1)
        xs1 = layer(xs)

        layer = layers.Grayscale(output_channels=3)
        xs2 = layer(xs)

        self.assertEqual(xs1.shape, [2, 4, 8, 1])
        self.assertEqual(xs2.shape, [2, 4, 8, 3])

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((4, 8, 3)),
            dtype=tf.float32,
        )

        layer = layers.Grayscale(output_channels=1)
        xs1 = layer(xs)

        layer = layers.Grayscale(output_channels=3)
        xs2 = layer(xs)

        self.assertEqual(xs1.shape, [4, 8, 1])
        self.assertEqual(xs2.shape, [4, 8, 3])

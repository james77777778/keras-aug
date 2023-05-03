import tensorflow as tf

from keras_aug import layers


class RandomGaussianBlurTest(tf.test.TestCase):
    def test_in_single_image(self):
        layer = layers.RandomGaussianBlur(kernel_size=(3, 7), factor=(0, 2))

        # RGB
        xs = tf.cast(
            tf.ones((8, 8, 3)),
            dtype=tf.float32,
        )

        xs = layer(xs)
        self.assertEqual(xs.shape, [8, 8, 3])

        # greyscale
        xs = tf.cast(
            tf.ones((8, 8, 1)),
            dtype=tf.float32,
        )

        xs = layer(xs)
        self.assertEqual(xs.shape, [8, 8, 1])

    def test_non_square_images(self):
        layer = layers.RandomGaussianBlur(kernel_size=(3, 7), factor=(0, 2))

        # RGB
        xs = tf.ones((2, 4, 8, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 4, 8, 3])

        # greyscale
        xs = tf.ones((2, 4, 8, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 4, 8, 1])

    def test_single_input_args(self):
        layer = layers.RandomGaussianBlur(kernel_size=7, factor=2)

        # RGB
        xs = tf.ones((2, 8, 8, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 8, 8, 3])

        # greyscale
        xs = tf.ones((2, 8, 8, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 8, 8, 1])

    def test_numerical(self):
        layer = layers.RandomGaussianBlur(kernel_size=3, factor=(1.0, 1.0))

        xs = tf.expand_dims(
            tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            axis=-1,
        )

        xs = tf.expand_dims(xs, axis=0)

        # Result expected to be identical to gaussian blur kernel of
        # size 3x3 and factor=1.0
        result = tf.expand_dims(
            tf.constant(
                [
                    [0.07511361, 0.1238414, 0.07511361],
                    [0.1238414, 0.20417996, 0.1238414],
                    [0.07511361, 0.1238414, 0.07511361],
                ]
            ),
            axis=-1,
        )
        result = tf.expand_dims(result, axis=0)
        xs = layer(xs)

        self.assertAllClose(xs, result)

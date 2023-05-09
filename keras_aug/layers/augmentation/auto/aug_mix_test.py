import tensorflow as tf

from keras_aug import layers


class AugMixTest(tf.test.TestCase):
    def test_in_single_image(self):
        layer = layers.AugMix(value_range=(0, 255))

        # RGB
        xs = tf.cast(tf.ones((8, 8, 3)), dtype=tf.float32)

        xs = layer(xs)
        self.assertEqual(xs.shape, [8, 8, 3])

        # greyscale
        xs = tf.cast(tf.ones((8, 8, 1)), dtype=tf.float32)

        xs = layer(xs)
        self.assertEqual(xs.shape, [8, 8, 1])

    def test_non_square_images(self):
        layer = layers.AugMix(value_range=(0, 255))

        # RGB
        xs = tf.ones((2, 4, 8, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 4, 8, 3])

        # greyscale
        xs = tf.ones((2, 4, 8, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 4, 8, 1])

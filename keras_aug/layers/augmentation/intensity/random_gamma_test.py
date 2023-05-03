import tensorflow as tf

from keras_aug import layers


class RandomGammaTest(tf.test.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "factor": (1.0 - 0.25, 1.0 + 0.25),
    }
    no_aug_args = {
        "value_range": (0, 255),
        "factor": (1.0, 1.0),
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        layer = layers.RandomGamma(**self.no_aug_args)

        output = layer(image)

        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.RandomGamma(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

        layer = layers.RandomGamma(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_sqrt_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape)
        args = self.no_aug_args.copy()
        args.update({"value_range": (0, 1), "factor": (0.5, 0.5)})
        layer = layers.RandomGamma(**args)

        output = layer(image)

        self.assertAllClose(tf.sqrt(image), output, atol=1e-5, rtol=1e-5)

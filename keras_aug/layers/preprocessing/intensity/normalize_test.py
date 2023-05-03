import tensorflow as tf

from keras_aug import layers


class NormalizeTest(tf.test.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
    no_aug_args = {
        "value_range": (0, 1),
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape)

        layer = layers.Normalize(**self.no_aug_args)
        output = layer(image)

        self.assertAllClose(image, output)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.Normalize(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_normalize_output(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        mean = tf.convert_to_tensor(self.regular_args["mean"])
        mean = tf.reshape(mean, shape=(1, 1, 1, 3))
        std = tf.convert_to_tensor(self.regular_args["std"])
        std = tf.reshape(std, shape=(1, 1, 1, 3))
        expected_output = (image - mean * 255.0) / (std * 255.0)

        layer = layers.Normalize(**self.regular_args)
        output = layer(image)

        self.assertAllClose(output, expected_output)

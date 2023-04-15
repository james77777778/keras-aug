import numpy as np
import tensorflow as tf

from keras_aug import augmentations


class NormalizeTest(tf.test.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }

    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.Normalize(**self.regular_args)
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_normalize_output(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        mean = tf.convert_to_tensor(self.regular_args["mean"])
        mean = tf.reshape(mean, shape=(1, 1, 1, 3))
        std = tf.convert_to_tensor(self.regular_args["std"])
        std = tf.reshape(std, shape=(1, 1, 1, 3))
        expected_output = (image - mean * 255.0) / (std * 255.0)

        layer = augmentations.Normalize(**self.regular_args)
        output = layer(image)

        self.assertAllClose(output, expected_output)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = augmentations.Normalize(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_with_unit8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = augmentations.Normalize(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_config_with_custom_name(self):
        layer = augmentations.Normalize(
            **self.regular_args,
            name="image_preproc",
        )
        config = layer.get_config()
        layer_1 = augmentations.Normalize.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_config(self):
        layer = augmentations.Normalize(**self.regular_args)
        config = layer.get_config()
        self.assertEqual(
            config["value_range"], self.regular_args["value_range"]
        )
        self.assertEqual(config["mean"], self.regular_args["mean"])
        self.assertEqual(config["std"], self.regular_args["std"])

    def test_output_dtypes(self):
        inputs = np.array(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype="float64"
        )
        layer = augmentations.Normalize(**self.regular_args)
        self.assertAllEqual(layer(inputs).dtype, "float32")

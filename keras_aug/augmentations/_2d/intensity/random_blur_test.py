import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_aug import augmentations


class RandomBlurTest(tf.test.TestCase, parameterized.TestCase):
    regular_args = {
        "factor": (3, 7),
    }
    no_aug_args = {
        "factor": (1, 1),
    }

    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.RandomBlur(**self.regular_args)
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_with_uint8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = augmentations.RandomBlur(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = augmentations.RandomBlur(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_independence_on_batched_images(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = augmentations.RandomBlur(**self.regular_args, seed=2023)

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_config_with_custom_name(self):
        layer = augmentations.RandomBlur(
            **self.regular_args,
            name="image_preproc",
        )
        config = layer.get_config()
        layer_1 = augmentations.RandomBlur.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_config(self):
        layer = augmentations.RandomBlur(**self.regular_args)

        config = layer.get_config()

        self.assertEqual(config["factor"], self.regular_args["factor"])

    def test_output_dtypes(self):
        inputs = np.array(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype="float64"
        )

        layer = augmentations.RandomBlur(**self.regular_args)

        self.assertAllEqual(layer(inputs).dtype, "float32")

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.RandomBlur(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = augmentations.RandomBlur(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = augmentations.RandomBlur(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

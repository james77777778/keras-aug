import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras_cv import core

from keras_aug import augmentations


class RandomJpegQualityTest(tf.test.TestCase, parameterized.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "factor": (0.75, 1.0),  # 25%
    }

    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.RandomJpegQuality(**self.regular_args)
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_with_uint8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = augmentations.RandomJpegQuality(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_independence_on_batched_images(self):
        image = tf.random.uniform((100, 100, 3)) * 255.0
        batched_images = tf.stack((image, image), axis=0)
        layer = augmentations.RandomJpegQuality(**self.regular_args, seed=2023)

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_config_with_custom_name(self):
        layer = augmentations.RandomJpegQuality(
            **self.regular_args,
            name="image_preproc",
        )
        config = layer.get_config()

        layer_1 = augmentations.RandomJpegQuality.from_config(config)

        self.assertEqual(layer_1.name, layer.name)

    def test_config(self):
        layer = augmentations.RandomJpegQuality(**self.regular_args)

        config = layer.get_config()

        self.assertEqual(
            config["value_range"], self.regular_args["value_range"]
        )
        self.assertTrue(isinstance(config["factor"], core.UniformFactorSampler))
        self.assertEqual(
            config["factor"].get_config()["lower"],
            self.regular_args["factor"][0],
        )
        self.assertEqual(
            config["factor"].get_config()["upper"],
            self.regular_args["factor"][1],
        )

    def test_output_dtypes(self):
        inputs = np.array(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype="float64"
        )

        layer = augmentations.RandomJpegQuality(**self.regular_args)

        self.assertAllEqual(layer(inputs).dtype, "float32")

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0
        args = self.regular_args.copy()
        args.update({"value_range": (0, 100)})
        layer = augmentations.RandomJpegQuality(**args)

        output = layer(image)

        self.assertNotAllClose(image, output)

    def test_in_single_image(self):
        layer = augmentations.RandomJpegQuality(**self.regular_args)
        # RGB
        xs = tf.cast(
            tf.ones((512, 512, 3)) * 255.0,
            dtype=tf.float32,
        )

        xs = layer(xs)

        self.assertEqual(xs.shape, [512, 512, 3])

        # greyscale
        xs = tf.cast(
            tf.ones((512, 512, 1)) * 255.0,
            dtype=tf.float32,
        )

        xs = layer(xs)

        self.assertEqual(xs.shape, [512, 512, 1])

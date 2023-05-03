import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers


class RandomBrightnessContrastTest(tf.test.TestCase, parameterized.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "brightness_factor": (0.6, 1.4),  # 40%
        "contrast_factor": (0.6, 1.4),  # 40%
    }
    no_aug_args = {
        "value_range": (0, 255),
        "brightness_factor": (1.0, 1.0),
        "contrast_factor": (1.0, 1.0),
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = layers.RandomBrightnessContrast(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.RandomBrightnessContrast(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = layers.RandomBrightnessContrast(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_max_brightness(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape, seed=2023) * 255.0
        args = self.no_aug_args.copy()
        args.update({"brightness_factor": (10000.0, 10000.0)})
        layer = layers.RandomBrightnessContrast(**args, seed=2023)

        output = layer(image)

        self.assertAllClose(output, tf.fill((4, 8, 8, 3), 255))

    def test_max_brightness_rescaled_value_range(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape, seed=2023)
        args = self.no_aug_args.copy()
        args.update(
            {"value_range": (0, 1), "brightness_factor": (10000.0, 10000.0)}
        )
        layer = layers.RandomBrightnessContrast(**args, seed=2023)

        output = layer(image)

        self.assertAllClose(output, tf.fill((4, 8, 8, 3), 1))

    def test_zero_brightness(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        args = self.no_aug_args.copy()
        args.update({"brightness_factor": (0.0, 0.0)})
        layer = layers.RandomBrightnessContrast(**args)

        output = layer(image)

        self.assertAllClose(output, tf.fill((4, 8, 8, 3), 0))

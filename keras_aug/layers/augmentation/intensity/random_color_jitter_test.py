import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers


class RandomColorJitterTest(tf.test.TestCase, parameterized.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "brightness_factor": (0.6, 1.4),  # 40%
        "contrast_factor": (0.6, 1.4),  # 40%
        "hue_factor": (-0.015, 0.015),  # 1.5%
        "saturation_factor": (0.6, 1.4),  # 40%
    }
    no_aug_args = {
        "value_range": (0, 255),
        "brightness_factor": (1.0, 1.0),
        "contrast_factor": (1.0, 1.0),
        "hue_factor": (0.0, 0.0),
        "saturation_factor": (1.0, 1.0),
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = layers.RandomColorJitter(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.RandomColorJitter(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = layers.RandomColorJitter(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_max_brightness(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape, seed=2023) * 255.0
        args = self.no_aug_args.copy()
        args.update({"brightness_factor": (10000.0, 10000.0)})
        layer = layers.RandomColorJitter(**args, seed=2023)

        output = layer(image)

        self.assertAllClose(output, tf.fill((4, 8, 8, 3), 255))

    def test_max_brightness_rescaled_value_range(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape, seed=2023)
        args = self.no_aug_args.copy()
        args.update(
            {"value_range": (0, 1), "brightness_factor": (10000.0, 10000.0)}
        )
        layer = layers.RandomColorJitter(**args, seed=2023)

        output = layer(image)

        self.assertAllClose(output, tf.fill((4, 8, 8, 3), 1))

    def test_zero_brightness(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        args = self.no_aug_args.copy()
        args.update({"brightness_factor": (0.0, 0.0)})
        layer = layers.RandomColorJitter(**args)

        output = layer(image)

        self.assertAllClose(output, tf.fill((4, 8, 8, 3), 0))

    def test_adjust_full_opposite_hue(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        args = self.no_aug_args.copy()
        args.update({"hue_factor": (0.0, 0.0)})

        layer = layers.RandomColorJitter(**args)
        output = layer(image)

        channel_max = tf.math.reduce_max(output, axis=-1)
        channel_min = tf.math.reduce_min(output, axis=-1)
        # Make sure the max and min channel are the same between input and
        # output. In the meantime, and channel will swap between each other.
        self.assertAllClose(
            channel_max,
            tf.math.reduce_max(image, axis=-1),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertAllClose(
            channel_min,
            tf.math.reduce_min(image, axis=-1),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_adjust_to_grayscale(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        args = self.no_aug_args.copy()
        args.update({"saturation_factor": (0.0, 0.0)})
        layer = layers.RandomColorJitter(**args)
        output = layer(image)

        channel_mean = tf.math.reduce_mean(output, axis=-1)
        channel_values = tf.unstack(output, axis=-1)
        # Make sure all the pixel has the same value among the channel dim,
        # which is a fully gray RGB.
        for channel_value in channel_values:
            self.assertAllClose(
                channel_mean, channel_value, atol=1e-5, rtol=1e-5
            )

    def test_adjust_to_full_saturation(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        args = self.no_aug_args.copy()
        args.update({"saturation_factor": (100.0, 100.0)})
        layer = layers.RandomColorJitter(**args)
        output = layer(image)

        channel_mean = tf.math.reduce_min(output, axis=-1)
        # Make sure at least one of the channel is 0.0 (fully saturated image)
        self.assertAllClose(channel_mean, tf.zeros((4, 8, 8)))

    @parameterized.named_parameters(("025", 0.25), ("05", 0.5))
    def test_adjusts_all_values_for_hue_factor(self, factor):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        args = self.no_aug_args.copy()
        args.update({"hue_factor": (0.0 - factor, 0.0 + factor)})
        layer = layers.RandomColorJitter(**args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    @parameterized.named_parameters(("025", 0.25), ("05", 0.5))
    def test_adjusts_all_values_for_saturation_factor(self, factor):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        args = self.no_aug_args.copy()
        args.update({"saturation_factor": (1.0 - factor, 1.0 + factor)})
        layer = layers.RandomColorJitter(**args)
        output = layer(image)
        self.assertNotAllClose(image, output)

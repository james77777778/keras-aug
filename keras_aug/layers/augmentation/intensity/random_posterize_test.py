import numpy as np
import tensorflow as tf

from keras_aug import layers


class RandomPosterizeTest(tf.test.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()

    def test_raises_error_on_invalid_bits_parameter(self):
        invalid_values = [-1, 0, 9, 24]
        for value in invalid_values:
            with self.assertRaises(ValueError):
                layers.RandomPosterize(value_range=(0, 1), factor=value)

    def test_single_image(self):
        bits = self._get_random_bits()
        dummy_input = self.rng.uniform(shape=(4, 4, 3), maxval=256)
        expected_output = self._calc_expected_output(dummy_input, bits=bits)

        layer = layers.RandomPosterize(
            value_range=(0, 255), factor=(bits, bits)
        )
        output = layer(dummy_input)

        self.assertAllEqual(output, expected_output)

    def _get_random_bits(self):
        return int(
            self.rng.uniform(shape=(), minval=1, maxval=9, dtype=tf.int32)
        )

    def test_single_image_rescaled(self):
        bits = self._get_random_bits()
        dummy_input = self.rng.uniform(shape=(4, 4, 3), maxval=1.0)
        expected_output = (
            self._calc_expected_output(dummy_input * 255, bits=bits) / 255
        )

        layer = layers.RandomPosterize(value_range=[0, 1], factor=(bits, bits))
        output = layer(dummy_input)

        self.assertAllClose(output, expected_output)

    def test_batched_input(self):
        bits = self._get_random_bits()
        dummy_input = self.rng.uniform(shape=(2, 4, 4, 3), maxval=256)

        expected_output = []
        for image in dummy_input:
            expected_output.append(self._calc_expected_output(image, bits=bits))
        expected_output = tf.stack(expected_output)

        layer = layers.RandomPosterize(
            value_range=(0, 255), factor=(bits, bits)
        )
        output = layer(dummy_input)

        self.assertAllEqual(output, expected_output)

    @staticmethod
    def _calc_expected_output(image, bits):
        """layers.RandomPosterize in numpy, based on Albumentations:
        The algorithm is basically:
        1. create a lookup table of all possible input pixel values to pixel
            values after RandomPosterize
        2. map each pixel in the input to created lookup table.

        Source:
            https://github.com/albumentations-team/albumentations/blob/89a675cbfb2b76f6be90e7049cd5211cb08169a5/albumentations/augmentations/functional.py#L407
        """  # noqa: E501
        dtype = image.dtype
        image = tf.cast(image, tf.uint8)

        lookup_table = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits) - 1)
        lookup_table &= mask

        return tf.cast(lookup_table[image], dtype)

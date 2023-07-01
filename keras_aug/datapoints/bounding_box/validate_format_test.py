"""
Most of these codes come from KerasCV.
"""
import tensorflow as tf

from keras_aug.datapoints import bounding_box


class ValidateTest(tf.test.TestCase):
    def test_raises_nondict(self):
        with self.assertRaisesRegex(
            ValueError, "Expected `bounding_boxes` to be a dictionary, got "
        ):
            bounding_box.validate_format(tf.ones((4, 3, 6)))

    def test_mismatch_dimensions(self):
        with self.assertRaisesRegex(
            ValueError,
            "Expected `boxes` and `classes` to have matching dimensions",
        ):
            bounding_box.validate_format(
                {"boxes": tf.ones((4, 3, 6)), "classes": tf.ones((4, 6))}
            )

    def test_bad_keys(self):
        with self.assertRaisesRegex(ValueError, "containing keys"):
            bounding_box.validate_format(
                {
                    "box": [1, 2, 3],
                    "class": [1234],
                }
            )

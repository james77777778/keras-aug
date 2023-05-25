import tensorflow as tf
from absl.testing import parameterized
from keras.backend import RandomGenerator

from keras_aug import core
from keras_aug.utils import augmentation as augmentation_utils


class AugmentationUtilsTest(tf.test.TestCase, parameterized.TestCase):
    def test_get_padding_position_invalid(self):
        with self.assertRaises(NotImplementedError):
            augmentation_utils.get_padding_position("hello")

    @parameterized.parameters(
        [
            "center",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "random",
        ]
    )
    def test_get_padding_position(self, position):
        _ = augmentation_utils.get_padding_position(position)

    @parameterized.named_parameters(
        *[
            ("center", ([1], [2], [3], [4]), ([1], [2], [3], [4]), "center"),
            (
                "top_left",
                ([1], [2], [3], [4]),
                ([0], [3], [0], [7]),
                "top_left",
            ),
            (
                "top_right",
                ([1], [2], [3], [4]),
                ([0], [3], [7], [0]),
                "top_right",
            ),
            (
                "bottom_left",
                ([1], [2], [3], [4]),
                ([3], [0], [0], [7]),
                "bottom_left",
            ),
            (
                "bottom_right",
                ([1], [2], [3], [4]),
                ([3], [0], [7], [0]),
                "bottom_right",
            ),
            ("random", ([1], [2], [3], [4]), None, "random"),
        ]
    )
    def test_get_position_params(self, values, expected_values, position_name):
        rng = RandomGenerator(2023)
        tops, bottoms, lefts, rights = values
        position = augmentation_utils.get_padding_position(position_name)

        tops, bottoms, lefts, rights = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, position, rng
        )
        if position_name == "random":
            return

        self.assertAllEqual((tops, bottoms, lefts, rights), expected_values)

    def test_is_factor_working_false(self):
        # int
        self.assertEqual(
            augmentation_utils.is_factor_working(0, not_working_value=0), False
        )
        # tuple
        self.assertEqual(
            augmentation_utils.is_factor_working((0, 0), not_working_value=0),
            False,
        )
        # core.ConstantFactorSampler
        self.assertEqual(
            augmentation_utils.is_factor_working(
                core.ConstantFactorSampler(0), not_working_value=0
            ),
            False,
        )
        # core.UniformFactorSampler
        self.assertEqual(
            augmentation_utils.is_factor_working(
                core.UniformFactorSampler(0, 0), not_working_value=0
            ),
            False,
        )

    def test_is_factor_working_true(self):
        # int
        self.assertEqual(
            augmentation_utils.is_factor_working(1, not_working_value=0), True
        )
        # tuple
        self.assertEqual(
            augmentation_utils.is_factor_working((1, 1), not_working_value=0),
            True,
        )
        # core.ConstantFactorSampler
        self.assertEqual(
            augmentation_utils.is_factor_working(
                core.ConstantFactorSampler(1), not_working_value=0
            ),
            True,
        )
        # core.UniformFactorSampler
        self.assertEqual(
            augmentation_utils.is_factor_working(
                core.UniformFactorSampler(0, 1), not_working_value=0
            ),
            True,
        )

    def test_expand_dict_dims(self):
        key = "factor"
        shape = [3, 3]
        transformation = {key: tf.random.uniform(shape=shape)}

        result = augmentation_utils.expand_dict_dims(transformation, axis=0)

        self.assertEqual(result[key].shape, [1] + [3, 3])

    def test_get_images_shape_dense(self):
        height = 5
        width = 6
        images = tf.random.uniform(shape=(2, 5, 6, 3))

        heights, widths = augmentation_utils.get_images_shape(images)

        self.assertAllEqual(heights, [[height], [height]])
        self.assertAllEqual(widths, [[width], [width]])

    def test_get_images_shape_ragged(self):
        images = tf.ragged.stack(
            [
                tf.ones((5, 5, 3)),
                tf.ones((8, 8, 3)),
            ]
        )

        heights, widths = augmentation_utils.get_images_shape(images)

        self.assertAllEqual(heights, [[5], [8]])
        self.assertAllEqual(widths, [[5], [8]])

    def test_cast_to_float16(self):
        inputs = {
            augmentation_utils.IMAGES: tf.random.uniform((2, 4, 4, 3)),
            augmentation_utils.LABELS: tf.random.uniform((2, 1)),
            augmentation_utils.BOUNDING_BOXES: {
                "boxes": tf.zeros((2, 2, 4)),
                "classes": tf.zeros((2, 2)),
            },
            augmentation_utils.SEGMENTATION_MASKS: tf.random.uniform(
                (2, 4, 4, 10)
            ),
            augmentation_utils.KEYPOINTS: tf.random.uniform((2, 4, 17)),
        }

        result = augmentation_utils.cast_to(inputs, dtype=tf.float16)

        self.assertDTypeEqual(result[augmentation_utils.IMAGES], tf.float16)
        self.assertDTypeEqual(result[augmentation_utils.LABELS], tf.float16)
        self.assertDTypeEqual(
            result[augmentation_utils.BOUNDING_BOXES]["boxes"], tf.float16
        )
        self.assertDTypeEqual(
            result[augmentation_utils.BOUNDING_BOXES]["classes"], tf.float16
        )
        self.assertDTypeEqual(result[augmentation_utils.KEYPOINTS], tf.float16)

    def test_cast_to_int8(self):
        inputs = {
            augmentation_utils.IMAGES: tf.random.uniform((2, 4, 4, 3)),
            augmentation_utils.LABELS: tf.random.uniform((2, 1)),
            augmentation_utils.BOUNDING_BOXES: {
                "boxes": tf.zeros((2, 2, 4)),
                "classes": tf.zeros((2, 2)),
            },
            augmentation_utils.SEGMENTATION_MASKS: tf.random.uniform(
                (2, 4, 4, 10)
            ),
            augmentation_utils.KEYPOINTS: tf.random.uniform((2, 4, 17)),
        }

        result = augmentation_utils.cast_to(inputs, dtype=tf.int8)

        self.assertDTypeEqual(result[augmentation_utils.IMAGES], tf.int8)
        self.assertDTypeEqual(result[augmentation_utils.LABELS], tf.int8)
        self.assertDTypeEqual(
            result[augmentation_utils.BOUNDING_BOXES]["boxes"], tf.int8
        )
        self.assertDTypeEqual(
            result[augmentation_utils.BOUNDING_BOXES]["classes"], tf.int8
        )
        self.assertDTypeEqual(result[augmentation_utils.KEYPOINTS], tf.int8)

    def test_compute_signature(self):
        inputs = {
            augmentation_utils.IMAGES: tf.random.uniform((2, 4, 4, 3)),
            augmentation_utils.LABELS: tf.random.uniform((2, 1)),
            augmentation_utils.BOUNDING_BOXES: {
                "boxes": tf.zeros((2, 2, 4)),
                "classes": tf.zeros((2, 2)),
            },
            augmentation_utils.SEGMENTATION_MASKS: tf.random.uniform(
                (2, 4, 4, 10)
            ),
            augmentation_utils.KEYPOINTS: tf.random.uniform((2, 4, 17)),
        }

        signatures = augmentation_utils.compute_signature(inputs, tf.float16)

        self.assertTrue(inputs.keys() == signatures.keys())

    def test_blend(self):
        ones = tf.ones(shape=(2, 4, 4, 3))
        twos = tf.ones(shape=(2, 4, 4, 3)) * 2
        ratios = tf.ones(shape=(2, 1, 1, 1)) * 0.3
        expected_result = ratios * ones + (1.0 - ratios) * twos

        result = augmentation_utils.blend(twos, ones, ratios)

        self.assertAllEqual(result, expected_result)

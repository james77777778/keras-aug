import inspect

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_aug.augmentations import _2d as augmentations
from keras_aug.utils.augmentation_utils import IMAGES
from keras_aug.utils.augmentation_utils import LABELS

SEED = 2025
TEST_CONFIGURATIONS = [
    (
        "CenterCrop",
        augmentations.CenterCrop,
        {"height": 2, "width": 2},
    ),
    (
        "PadIfNeeded",
        augmentations.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
    ),
    (
        "RandomAffine",
        augmentations.RandomAffine,
        {
            "rotation_factor": 10,
            "translation_height_factor": 0.1,
            "translation_width_factor": 0.1,
            "zoom_height_factor": 0.1,
            "zoom_width_factor": 0.1,
            "shear_height_factor": 0.1,
            "shear_width_factor": 0.1,
        },
    ),
    (
        "RandomCropAndResize",
        augmentations.RandomCropAndResize,
        {
            "height": 2,
            "width": 2,
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    (
        "ResizeAndPad",
        augmentations.ResizeAndPad,
        {"height": 2, "width": 2},
    ),
    (
        "ResizeByLongestSide",
        augmentations.ResizeByLongestSide,
        {"max_size": [2]},
    ),
    (
        "ResizeBySmallestSide",
        augmentations.ResizeBySmallestSide,
        {"min_size": [2]},
    ),
    (
        "CLAHE",
        augmentations.CLAHE,
        {"value_range": (0, 255), "factor": (2, 10), "tile_grid_size": (4, 4)},
    ),
    ("Normalize", augmentations.Normalize, {"value_range": (0, 255)}),
    ("RandomBlur", augmentations.RandomBlur, {"factor": (3, 7)}),
    (
        "RandomBrightnessContrast",
        augmentations.RandomBrightnessContrast,
        {
            "value_range": (0, 255),
            "brightness_factor": 0.1,
            "contrast_factor": 0.1,
        },
    ),
    (
        "RandomColorJitter",
        augmentations.RandomColorJitter,
        {
            "value_range": (0, 255),
            "brightness_factor": 0.1,
            "contrast_factor": 0.1,
            "saturation_factor": 0.1,
            "hue_factor": 0.1,
        },
    ),
    (
        "RandomGamma",
        augmentations.RandomGamma,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomHSV",
        augmentations.RandomHSV,
        {
            "value_range": (0, 255),
            "hue_factor": 0.1,
            "saturation_factor": 0.1,
            "value_factor": 0.1,
        },
    ),
    (
        "RandomJpegQuality",
        augmentations.RandomJpegQuality,
        {
            "value_range": (0, 255),
            "factor": (75, 100),
        },
    ),
    (
        "MosaicYOLOV8",
        augmentations.MosaicYOLOV8,
        {
            "height": 100,
            "width": 100,
        },
    ),
    (
        "ChannelDropout",
        augmentations.ChannelDropout,
        {},
    ),
]


class WithMixedPrecisionTest(tf.test.TestCase, parameterized.TestCase):
    def test_all_2d_aug_layers_included(self):
        base_cls = augmentations.VectorizedBaseRandomLayer
        all_2d_aug_layers = inspect.getmembers(
            augmentations,
            predicate=inspect.isclass,
        )
        all_2d_aug_layers = [
            item
            for item in all_2d_aug_layers
            if issubclass(item[1], base_cls) and item[1] is not base_cls
        ]
        all_2d_aug_layer_names = set(item[0] for item in all_2d_aug_layers)
        test_configuration_names = set(item[0] for item in TEST_CONFIGURATIONS)

        for name in all_2d_aug_layer_names:
            self.assertIn(
                name,
                test_configuration_names,
                msg=f"{name} not found in TEST_CONFIGURATIONS",
            )

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_in_mixed_precision(self, layer_cls, args):
        keras.mixed_precision.set_global_policy("mixed_float16")
        images = tf.cast(
            tf.random.uniform(shape=(4, 32, 32, 3), seed=SEED) * 255.0,
            dtype=tf.float64,
        )
        labels = tf.cast(
            tf.random.uniform(shape=(4, 1), seed=SEED) * 10.0, dtype=tf.float64
        )
        layer = layer_cls(**args)
        layer({IMAGES: images, LABELS: labels})

    @classmethod
    def tearDownClass(cls) -> None:
        # Do not affect other tests
        keras.mixed_precision.set_global_policy("float32")

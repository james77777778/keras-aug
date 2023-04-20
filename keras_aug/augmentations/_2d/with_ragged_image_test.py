import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug.augmentations import _2d as augmentations
from keras_aug.utils import augmentation_utils

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
    ("CLAHE", augmentations.CLAHE, {"value_range": (0, 255)}),
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
        "MixUp",
        augmentations.MixUp,
        {},
    ),
    (
        "MosaicYOLOV8",
        augmentations.MosaicYOLOV8,
        {
            "height": 2,
            "width": 2,
        },
    ),
    (
        "ChannelDropout",
        augmentations.ChannelDropout,
        {},
    ),
    (
        "RandomApply",
        augmentations.RandomApply,
        {"layer": augmentations.ChannelDropout()},
    ),
]


class RaggedImageTest(tf.test.TestCase, parameterized.TestCase):
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
    def test_preserves_ragged_status(self, layer_cls, args):
        layer = layer_cls(**args)
        # MixUp needs two same shape image
        if layer_cls == augmentations.MixUp:
            images = tf.ragged.stack(
                [
                    tf.ones((8, 8, 3)),
                    tf.ones((8, 8, 3)),
                ]
            )
        else:
            images = tf.ragged.stack(
                [
                    tf.ones((5, 5, 3)),
                    tf.ones((8, 8, 3)),
                ]
            )
        labels = tf.ragged.stack(
            [
                tf.ones((1,)),
                tf.ones((1,)),
            ]
        )
        inputs = {
            augmentation_utils.IMAGES: images,
            augmentation_utils.LABELS: labels,
        }

        outputs = layer(inputs)

        self.assertTrue(
            isinstance(outputs[augmentation_utils.IMAGES], tf.RaggedTensor)
        )

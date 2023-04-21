import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug.augmentation import _2d as augmentation
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

TEST_CONFIGURATIONS = [
    (
        "PadIfNeeded",
        augmentation.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
    ),
    (
        "RandomAffine",
        augmentation.RandomAffine,
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
        "ResizeByLongestSide",
        augmentation.ResizeByLongestSide,
        {"max_size": [2]},
    ),
    (
        "ResizeBySmallestSide",
        augmentation.ResizeBySmallestSide,
        {"min_size": [2]},
    ),
    ("CLAHE", augmentation.CLAHE, {"value_range": (0, 255)}),
    ("Normalize", augmentation.Normalize, {"value_range": (0, 255)}),
    ("RandomBlur", augmentation.RandomBlur, {"factor": (3, 7)}),
    (
        "RandomBrightnessContrast",
        augmentation.RandomBrightnessContrast,
        {
            "value_range": (0, 255),
            "brightness_factor": 0.1,
            "contrast_factor": 0.1,
        },
    ),
    (
        "RandomColorJitter",
        augmentation.RandomColorJitter,
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
        augmentation.RandomGamma,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomHSV",
        augmentation.RandomHSV,
        {
            "value_range": (0, 255),
            "hue_factor": 0.1,
            "saturation_factor": 0.1,
            "value_factor": 0.1,
        },
    ),
    (
        "RandomJpegQuality",
        augmentation.RandomJpegQuality,
        {
            "value_range": (0, 255),
            "factor": (75, 100),
        },
    ),
    (
        "Rescaling",
        augmentation.Rescaling,
        {"scale": 1.0 / 255.0},
    ),
    (
        "MixUp",
        augmentation.MixUp,
        {},
    ),
    (
        "ChannelDropout",
        augmentation.ChannelDropout,
        {},
    ),
    (
        "RandomApply",
        augmentation.RandomApply,
        {"layer": augmentation.ChannelDropout()},
    ),
]

FORCE_DENSE_IMAGES_LAYERS = [
    (
        "CenterCrop",
        augmentation.CenterCrop,
        {"height": 2, "width": 2},
    ),
    (
        "RandomCropAndResize",
        augmentation.RandomCropAndResize,
        {
            "height": 2,
            "width": 2,
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    (
        "ResizeAndPad",
        augmentation.ResizeAndPad,
        {"height": 2, "width": 2},
    ),
    (
        "MosaicYOLOV8",
        augmentation.MosaicYOLOV8,
        {
            "height": 2,
            "width": 2,
        },
    ),
]


class WithRaggedImageTest(tf.test.TestCase, parameterized.TestCase):
    def test_all_2d_aug_layers_included(self):
        base_cls = augmentation.VectorizedBaseRandomLayer
        all_2d_aug_layers = inspect.getmembers(
            augmentation,
            predicate=inspect.isclass,
        )
        all_2d_aug_layers = [
            item
            for item in all_2d_aug_layers
            if issubclass(item[1], base_cls) and item[1] is not base_cls
        ]
        all_2d_aug_layer_names = set(item[0] for item in all_2d_aug_layers)
        test_conf_names = set(item[0] for item in TEST_CONFIGURATIONS)
        force_dense_names = set(item[0] for item in FORCE_DENSE_IMAGES_LAYERS)
        all_test_conf_names = test_conf_names.union(force_dense_names)

        for name in all_2d_aug_layer_names:
            self.assertIn(
                name,
                all_test_conf_names,
                msg=f"{name} not found in TEST_CONFIGURATIONS",
            )

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_preserves_ragged_status(self, layer_cls, args):
        layer = layer_cls(**args)
        # MixUp needs two same shape image
        if layer_cls == augmentation.MixUp:
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
        inputs = {IMAGES: images, LABELS: labels}

        outputs = layer(inputs)

        self.assertTrue(isinstance(outputs[IMAGES], tf.RaggedTensor))

    @parameterized.named_parameters(*FORCE_DENSE_IMAGES_LAYERS)
    def test_force_dense_images(self, layer_cls, args):
        layer = layer_cls(**args)
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
        inputs = {IMAGES: images, LABELS: labels}

        outputs = layer(inputs)

        self.assertTrue(isinstance(outputs[IMAGES], tf.Tensor))

import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug.augmentation import _2d as augmentation
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

TEST_CONFIGURATIONS = [
    (
        "RandAugment",
        augmentation.RandAugment,
        {"value_range": (0, 255), "seed": 2023},
    ),
    (
        "CenterCrop",
        augmentation.CenterCrop,
        {"height": 2, "width": 2},
    ),
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
    ("RandomCrop", augmentation.RandomCrop, {"height": 2, "width": 2}),
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
    ("RandomFlip", augmentation.RandomFlip, {"mode": "horizontal"}),
    ("RandomRotate", augmentation.RandomRotate, {"factor": 10}),
    (
        "Resize",
        augmentation.Resize,
        {"height": 2, "width": 2},
    ),
    (
        "ResizeAndCrop",
        augmentation.ResizeAndCrop,
        {"height": 2, "width": 2},
    ),
    (
        "ResizeAndPad",
        augmentation.ResizeAndPad,
        {"height": 2, "width": 2},
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
    ("AutoContrast", augmentation.AutoContrast, {"value_range": (0, 255)}),
    ("ChannelShuffle", augmentation.ChannelShuffle, {"groups": 3}),
    ("Equalize", augmentation.Equalize, {"value_range": (0, 255)}),
    ("Grayscale", augmentation.Grayscale, {"output_channels": 3}),
    ("Invert", augmentation.Invert, {"value_range": (0, 255)}),
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
        "RandomChannelShift",
        augmentation.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomCLAHE",
        augmentation.RandomCLAHE,
        {"value_range": (0, 255), "factor": (2, 10), "tile_grid_size": (4, 4)},
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
        "RandomPosterize",
        augmentation.RandomPosterize,
        {"value_range": (0, 255), "factor": (5, 8)},
    ),
    (
        "RandomSharpness",
        augmentation.RandomSharpness,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomSolarize",
        augmentation.RandomSolarize,
        {
            "value_range": (0, 255),
            "threshold_factor": 10,
            "addition_factor": 10,
        },
    ),
    (
        "Rescale",
        augmentation.Rescale,
        {"scale": 1.0 / 255.0},
    ),
    (
        "MixUp",
        augmentation.MixUp,
        {},
    ),
    (
        "MosaicYOLOV8",
        augmentation.MosaicYOLOV8,
        {
            "height": 100,
            "width": 100,
        },
    ),
    (
        "RandomChannelDropout",
        augmentation.RandomChannelDropout,
        {},
    ),
    (
        "RandomCutout",
        augmentation.RandomCutout,
        {"height_factor": 0.3, "width_factor": 0.3},
    ),
    ("Identity", augmentation.Identity, {}),
    (
        "RandomApply",
        augmentation.RandomApply,
        {"layer": augmentation.RandomChannelDropout()},
    ),
]

NO_XLA_SUPPORT_LAYERS = [
    augmentation.RandAugment,
    augmentation.RandomAffine,  # tf.raw_ops.ImageProjectiveTransformV3
    augmentation.RandomCrop,  # tf.image.crop_and_resize
    augmentation.RandomCropAndResize,  # tf.image.crop_and_resize
    augmentation.RandomRotate,  # tf.raw_ops.ImageProjectiveTransformV3
    augmentation.ResizeAndCrop,  # tf.image.crop_and_resize
    augmentation.ResizeByLongestSide,  # tf.image.resize
    augmentation.ResizeBySmallestSide,  # tf.image.resize
    augmentation.RandomBlur,  # tf.map_fn
    augmentation.RandomJpegQuality,  # tf.image.adjust_jpeg_quality
    augmentation.MixUp,  # tf.random.gamma / tf.random.stateless_gamma
    augmentation.MosaicYOLOV8,  # tf.map_fn
]

SKIP_XLA_TEST_LAYERS = [
    augmentation.RandAugment,  # too slow to compile
]


class GraphModeTest(tf.test.TestCase, parameterized.TestCase):
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
        test_configuration_names = set(item[0] for item in TEST_CONFIGURATIONS)

        for name in all_2d_aug_layer_names:
            self.assertIn(
                name,
                test_configuration_names,
                msg=f"{name} not found in TEST_CONFIGURATIONS",
            )

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_in_graph_mode(self, layer_cls, args):
        images = tf.random.uniform(shape=(1, 8, 8, 3)) * 255.0
        labels = tf.random.uniform(shape=(1, 1)) * 10.0
        layer = layer_cls(**args)

        @tf.function
        def fn(inputs):
            layer(inputs)

        fn({IMAGES: images, LABELS: labels})

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_in_xla_mode(self, layer_cls, args):
        if layer_cls in SKIP_XLA_TEST_LAYERS:
            return
        images = tf.random.uniform(shape=(1, 8, 8, 3)) * 255.0
        labels = tf.random.uniform(shape=(1, 1)) * 10.0
        layer = layer_cls(**args)

        @tf.function(jit_compile=True)
        def fn(inputs):
            layer(inputs)

        if layer_cls not in NO_XLA_SUPPORT_LAYERS:
            fn({IMAGES: images, LABELS: labels})
        else:
            with self.assertRaises(tf.errors.InvalidArgumentError):
                fn({IMAGES: images, LABELS: labels})

import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers
from keras_aug.layers import augmentation
from keras_aug.layers import preprocessing
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

TEST_CONFIGURATIONS = [
    ("AugMix", layers.AugMix, {"value_range": (0, 255)}),
    (
        "RandAugment",
        layers.RandAugment,
        {"value_range": (0, 255), "seed": 2023},
    ),
    (
        "TrivialAugmentWide",
        layers.TrivialAugmentWide,
        {"value_range": (0, 255)},
    ),
    (
        "RandomAffine",
        layers.RandomAffine,
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
    ("RandomCrop", layers.RandomCrop, {"height": 2, "width": 2}),
    (
        "RandomCropAndResize",
        layers.RandomCropAndResize,
        {
            "height": 2,
            "width": 2,
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    ("RandomFlip", layers.RandomFlip, {"mode": "horizontal"}),
    ("RandomResize", layers.RandomResize, {"heights": [2]}),
    ("RandomRotate", layers.RandomRotate, {"factor": 10}),
    (
        "RandomZoomAndCrop",
        layers.RandomZoomAndCrop,
        {"height": 2, "width": 2, "scale_factor": (0.8, 1.25)},
    ),
    ("ChannelShuffle", layers.ChannelShuffle, {"groups": 3}),
    ("RandomBlur", layers.RandomBlur, {"factor": (3, 7)}),
    (
        "RandomChannelShift",
        layers.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomCLAHE",
        layers.RandomCLAHE,
        {"value_range": (0, 255), "factor": (2, 10), "tile_grid_size": (4, 4)},
    ),
    (
        "RandomColorJitter",
        layers.RandomColorJitter,
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
        layers.RandomGamma,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomGaussianBlur",
        layers.RandomGaussianBlur,
        {"kernel_size": 3, "factor": 2.0},
    ),
    (
        "RandomHSV",
        layers.RandomHSV,
        {
            "value_range": (0, 255),
            "hue_factor": 0.1,
            "saturation_factor": 0.1,
            "value_factor": 0.1,
        },
    ),
    (
        "RandomJpegQuality",
        layers.RandomJpegQuality,
        {
            "value_range": (0, 255),
            "factor": (75, 100),
        },
    ),
    (
        "RandomPosterize",
        layers.RandomPosterize,
        {"value_range": (0, 255), "factor": (5, 8)},
    ),
    (
        "RandomSharpness",
        layers.RandomSharpness,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomSolarize",
        layers.RandomSolarize,
        {
            "value_range": (0, 255),
            "threshold_factor": 10,
            "addition_factor": 10,
        },
    ),
    (
        "CutMix",
        layers.CutMix,
        {"alpha": 1.0},
    ),
    (
        "MixUp",
        layers.MixUp,
        {},
    ),
    (
        "Mosaic",
        layers.Mosaic,
        {
            "height": 100,
            "width": 100,
        },
    ),
    (
        "RandomChannelDropout",
        layers.RandomChannelDropout,
        {},
    ),
    (
        "RandomCutout",
        layers.RandomCutout,
        {"height_factor": 0.3, "width_factor": 0.3},
    ),
    (
        "RandomErase",
        layers.RandomErase,
        {"area_factor": (0.02, 0.4), "aspect_ratio_factor": (0.3, 1.0 / 0.3)},
    ),
    (
        "RandomGridMask",
        layers.RandomGridMask,
        {
            "size_factor": (0.5, 1.0),
            "ratio_factor": (0.6, 0.6),
            "rotation_factor": (-10, 10),
        },
    ),
    (
        "RandomApply",
        layers.RandomApply,
        {"layer": layers.RandomChannelDropout()},
    ),
    (
        "RandomChoice",
        layers.RandomChoice,
        {
            "layers": [
                layers.RandomChannelDropout(),
                layers.RandomChannelDropout(),
            ]
        },
    ),
    (
        "RepeatedAugment",
        layers.RepeatedAugment,
        {
            "layers": [
                layers.RandomColorJitter(
                    value_range=(0, 255), brightness_factor=(1.5, 1.5)
                ),
                layers.RandomColorJitter(
                    value_range=(0, 255), contrast_factor=(1.5, 1.5)
                ),
            ]
        },
    ),
    (
        "CenterCrop",
        layers.CenterCrop,
        {"height": 2, "width": 2},
    ),
    (
        "PadIfNeeded",
        layers.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
    ),
    (
        "Resize",
        layers.Resize,
        {"height": 2, "width": 2},
    ),
    ("AutoContrast", layers.AutoContrast, {"value_range": (0, 255)}),
    ("Equalize", layers.Equalize, {"value_range": (0, 255)}),
    ("Grayscale", layers.Grayscale, {"output_channels": 3}),
    ("Invert", layers.Invert, {"value_range": (0, 255)}),
    ("Normalize", layers.Normalize, {"value_range": (0, 255)}),
    (
        "Rescale",
        layers.Rescale,
        {"scale": 1.0 / 255.0},
    ),
    ("Identity", layers.Identity, {}),
]

NO_XLA_SUPPORT_LAYERS = [
    layers.AugMix,
    layers.RandAugment,
    layers.TrivialAugmentWide,
    layers.RandomAffine,  # tf.raw_ops.ImageProjectiveTransformV3
    layers.RandomCrop,  # tf.image.crop_and_resize
    layers.RandomCropAndResize,  # tf.image.crop_and_resize
    layers.RandomResize,  # tf.image.resize
    layers.RandomRotate,  # tf.raw_ops.ImageProjectiveTransformV3
    layers.RandomZoomAndCrop,  # tf.image.resize
    layers.RandomBlur,  # tf.map_fn
    layers.RandomJpegQuality,  # tf.image.adjust_jpeg_quality
    layers.Mosaic,  # tf.map_fn
    layers.RandomGridMask,  # tf.raw_ops.ImageProjectiveTransformV3
    layers.RepeatedAugment,  # tf.random.state_less.shuffle
    layers.Equalize,  # tf.histogram_fixed_width
]

SKIP_XLA_TEST_LAYERS = [
    layers.AugMix,  # too slow to compile
    layers.RandAugment,  # too slow to compile
    layers.TrivialAugmentWide,  # too slow to compile
    layers.RandomColorJitter,  # too slow to compile
    layers.Equalize,  # too slow to compile
]


class GraphModeTest(tf.test.TestCase, parameterized.TestCase):
    def test_all_2d_aug_layers_included(self):
        base_cls = layers.VectorizedBaseRandomLayer
        all_2d_aug_layers = inspect.getmembers(
            augmentation,
            predicate=inspect.isclass,
        ) + inspect.getmembers(
            preprocessing,
            predicate=inspect.isclass,
        )
        all_2d_aug_layers = [
            item for item in all_2d_aug_layers if issubclass(item[1], base_cls)
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

import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers
from keras_aug.core import ConstantFactorSampler
from keras_aug.core import UniformFactorSampler
from keras_aug.layers import augmentation
from keras_aug.layers import preprocessing

GENERAL_TESTS = [
    (
        "AugMix",
        layers.AugMix,
        {"value_range": (0, 255)},
    ),
    (
        "RandAugment",
        layers.RandAugment,
        {"value_range": (0, 255)},
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
    (
        "RandomCrop",
        layers.RandomCrop,
        {"height": 2, "width": 2},
    ),
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
    (
        "RandomFlip",
        layers.RandomFlip,
        {"mode": "horizontal"},
    ),
    (
        "RandomResize",
        layers.RandomResize,
        {"heights": [2]},
    ),
    (
        "RandomRotate",
        layers.RandomRotate,
        {"factor": 10},
    ),
    (
        "RandomZoomAndCrop",
        layers.RandomZoomAndCrop,
        {"height": 2, "width": 2, "scale_factor": (0.8, 1.25)},
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
        "ChannelShuffle",
        layers.ChannelShuffle,
        {"groups": 3},
    ),
    (
        "RandomBlur",
        layers.RandomBlur,
        {"factor": (3, 7)},
    ),
    (
        "RandomChannelShift",
        layers.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomCLAHE",
        layers.RandomCLAHE,
        {"value_range": (0, 255)},
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
            "height": 2,
            "width": 2,
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
        "Resize",
        layers.Resize,
        {"height": 2, "width": 2},
    ),
    (
        "AutoContrast",
        layers.AutoContrast,
        {"value_range": (0, 255)},
    ),
    (
        "Equalize",
        layers.Equalize,
        {"value_range": (0, 255)},
    ),
    (
        "Grayscale",
        layers.Grayscale,
        {"output_channels": 3},
    ),
    (
        "Identity",
        layers.Identity,
        {},
    ),
    (
        "Invert",
        layers.Invert,
        {"value_range": (0, 255)},
    ),
    (
        "Normalize",
        layers.Normalize,
        {"value_range": (0, 255)},
    ),
    (
        "Rescale",
        layers.Rescale,
        {"scale": 1.0 / 255.0},
    ),
    (
        "SanitizeBoundingBox",
        layers.SanitizeBoundingBox,
        {"min_size": 10, "bounding_box_format": "xyxy"},
    ),
]


class ConfigTest(tf.test.TestCase, parameterized.TestCase):
    def test_all_2d_aug_layers_included(self):
        base_cls = layers.VectorizedBaseRandomLayer
        cls_spaces = [augmentation, preprocessing]
        all_2d_aug_layers = []
        for cls_space in cls_spaces:
            all_2d_aug_layers.extend(
                inspect.getmembers(cls_space, predicate=inspect.isclass)
            )
        all_2d_aug_layer_names = set(
            item[0]
            for item in all_2d_aug_layers
            if issubclass(item[1], base_cls)
        )

        general_names = set(item[0] for item in GENERAL_TESTS)
        all_test_names = general_names

        for name in all_2d_aug_layer_names:
            self.assertIn(name, all_test_names, msg=f"{name} not found")

    @parameterized.named_parameters(*GENERAL_TESTS)
    def test_config(self, layer_cls, args):
        layer = layer_cls(**args)

        config = layer.get_config()

        for key in args.keys():
            if isinstance(config[key], UniformFactorSampler):
                self.assertTrue(isinstance(config[key], UniformFactorSampler))
            elif isinstance(config[key], ConstantFactorSampler):
                self.assertTrue(isinstance(config[key], ConstantFactorSampler))
            else:
                self.assertEqual(config[key], args[key])

    @parameterized.named_parameters(*GENERAL_TESTS)
    def test_config_with_custom_name(self, layer_cls, args):
        layer = layer_cls(**args, name="image_preproc")
        config = layer.get_config()

        layer_1 = layer_cls.from_config(config)

        self.assertEqual(layer_1.name, layer.name)

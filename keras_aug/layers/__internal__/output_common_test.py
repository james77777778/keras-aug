import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers
from keras_aug.layers import augmentation
from keras_aug.layers import preprocessing
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

CONSISTENT_OUTPUTS_LAYERS = [
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
    ("RandomFlip", layers.RandomFlip, {"mode": "horizontal"}),
    ("RandomRotate", layers.RandomRotate, {"factor": 10}),
    ("ChannelShuffle", layers.ChannelShuffle, {"groups": 3}),
    ("RandomBlur", layers.RandomBlur, {"factor": (3, 7)}),
    (
        "RandomChannelShift",
        layers.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    ("RandomCLAHE", layers.RandomCLAHE, {"value_range": (0, 255)}),
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
        "PadIfNeeded",
        layers.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
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

FORCE_DENSE_IMAGES_LAYERS = [
    (
        "CenterCrop",
        layers.CenterCrop,
        {"height": 2, "width": 2},
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
    ("RandomResize", layers.RandomResize, {"heights": [2]}),
    (
        "RandomZoomAndCrop",
        layers.RandomZoomAndCrop,
        {"height": 2, "width": 2, "scale_factor": (0.8, 1.25)},
    ),
    (
        "Resize",
        layers.Resize,
        {"height": 2, "width": 2},
    ),
    (
        "Mosaic",
        layers.Mosaic,
        {
            "height": 2,
            "width": 2,
        },
    ),
]

NO_RAGGED_IMAGES_SUPPORT = [
    ("CutMix", layers.CutMix, {"alpha": 1.0}),
    ("MixUp", layers.MixUp, {}),
]


class WithRaggedImageTest(tf.test.TestCase, parameterized.TestCase):
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
        cosistent_names = set(item[0] for item in CONSISTENT_OUTPUTS_LAYERS)
        force_dense_names = set(item[0] for item in FORCE_DENSE_IMAGES_LAYERS)
        no_ragged_names = set(item[0] for item in NO_RAGGED_IMAGES_SUPPORT)
        all_test_conf_names = cosistent_names.union(force_dense_names).union(
            no_ragged_names
        )

        for name in all_2d_aug_layer_names:
            self.assertIn(
                name,
                all_test_conf_names,
                msg=f"{name} not found in TEST_CONFIGURATIONS",
            )

    @parameterized.named_parameters(*CONSISTENT_OUTPUTS_LAYERS)
    def test_preserves_ragged_status(self, layer_cls, args):
        layer = layer_cls(**args)
        # MixUp needs two same shape image
        if layer_cls == layers.MixUp:
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

import inspect

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_aug import layers
from keras_aug.layers import augmentation
from keras_aug.layers import preprocessing
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

TEST_CONFIGURATIONS = [
    ("AugMix", layers.AugMix, {"value_range": (0, 255)}),
    ("RandAugment", layers.RandAugment, {"value_range": (0, 255)}),
    (
        "TrivialAugmentWide",
        layers.TrivialAugmentWide,
        {
            "value_range": (0, 255),
            "use_geometry": False,
        },
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
    ("Equalize", layers.Equalize, {"value_range": (0, 255)}),
    ("Grayscale", layers.Grayscale, {"output_channels": 3}),
    ("Invert", layers.Invert, {"value_range": (0, 255)}),
    ("Normalize", layers.Normalize, {"value_range": (0, 255)}),
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
    (
        "Rescale",
        layers.Rescale,
        {"scale": 1.0 / 255.0},
    ),
    ("Identity", layers.Identity, {}),
]

BUILD_IN_RUNTIME = [
    (
        "RandomApply",
        layers.RandomApply,
        {"name": "layer", "value": "single", "args": {}},
        layers.RandomChannelDropout,
        {},
    ),
    (
        "RandomApplyBatchwise",
        layers.RandomApply,
        {"name": "layer", "value": "single", "args": {"batchwise": True}},
        layers.RandomChannelDropout,
        {},
    ),
    (
        "RandomChoice",
        layers.RandomChoice,
        {"name": "layers", "value": "multiple", "args": {}},
        layers.RandomChannelDropout,
        {},
    ),
    (
        "RandomChoiceBatchwise",
        layers.RandomChoice,
        {"name": "layers", "value": "multiple", "args": {"batchwise": True}},
        layers.RandomChannelDropout,
        {},
    ),
]


class WithMixedPrecisionTest(tf.test.TestCase, parameterized.TestCase):
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
        test_configs = set(item[0] for item in TEST_CONFIGURATIONS)
        build_in_runtime = set(item[0] for item in BUILD_IN_RUNTIME)
        all_test_conf_names = test_configs.union(build_in_runtime)

        for name in all_2d_aug_layer_names:
            self.assertIn(
                name,
                all_test_conf_names,
                msg=f"{name} not found in TEST_CONFIGURATIONS",
            )

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_can_run_in_mixed_precision(self, layer_cls, args):
        keras.mixed_precision.set_global_policy("mixed_float16")
        images = tf.cast(
            tf.random.uniform(shape=(2, 32, 32, 3)) * 255.0,
            dtype=tf.float64,
        )
        labels = tf.cast(
            tf.random.uniform(shape=(2, 1)) * 10.0, dtype=tf.float64
        )
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [[0, 1], [2]],
                dtype=tf.float32,
            ),
        }
        try:
            layer = layer_cls(**args, bounding_box_format="xyxy")
            has_bounding_boxes = True
        except TypeError:
            layer = layer_cls(**args)
            has_bounding_boxes = False

        if has_bounding_boxes:
            result = layer(
                {IMAGES: images, LABELS: labels, BOUNDING_BOXES: bounding_boxes}
            )
            self.assertDTypeEqual(result[IMAGES], tf.float16)
        else:
            result = layer({IMAGES: images, LABELS: labels})
            self.assertDTypeEqual(result[IMAGES], tf.float16)

    @parameterized.named_parameters(*BUILD_IN_RUNTIME)
    def test_can_run_in_mixed_precision_build_in_runtime(
        self, layer_cls, args, build_layer, build_args
    ):
        keras.mixed_precision.set_global_policy("mixed_float16")
        images = tf.cast(
            tf.random.uniform(shape=(4, 32, 32, 3)) * 255.0,
            dtype=tf.float64,
        )
        labels = tf.cast(
            tf.random.uniform(shape=(4, 1)) * 10.0, dtype=tf.float64
        )
        name = args["name"]
        value = args["value"]
        other_args = args["args"]
        if value == "single":
            build_layers = build_layer(**build_args)
        elif value == "multiple":
            build_layers = [
                build_layer(**build_args),
                build_layer(**build_args),
                build_layer(**build_args),
            ]
        else:
            raise NotImplementedError()
        args = {name: build_layers, **other_args}
        layer = layer_cls(**args)
        layer({IMAGES: images, LABELS: labels})

    @classmethod
    def tearDownClass(cls) -> None:
        # Do not affect other tests
        keras.mixed_precision.set_global_policy("float32")

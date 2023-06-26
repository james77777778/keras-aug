import inspect

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers
from keras_aug.layers import augmentation
from keras_aug.layers import preprocessing
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

#   (
#       name,
#       layer_cls,
#       args,
#       is_xla_compatible,
#   )
# all configurations should be expanded for readability
GENERAL_TESTS = [
    (
        "AugMix",
        layers.AugMix,
        {"value_range": (0, 255)},
        False,  # containing invalid operations
    ),
    (
        "RandAugment",
        layers.RandAugment,
        {"value_range": (0, 255), "seed": 2023},
        False,  # containing invalid operations
    ),
    (
        "TrivialAugmentWide",
        layers.TrivialAugmentWide,
        {"value_range": (0, 255)},
        False,  # containing invalid operations
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
        False,  # tf.raw_ops.ImageProjectiveTransformV3
    ),
    (
        "RandomCrop",
        layers.RandomCrop,
        {"height": 2, "width": 2},
        False,  # tf.image.crop_and_resize
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
        False,  # tf.image.crop_and_resize
    ),
    (
        "RandomFlip",
        layers.RandomFlip,
        {"mode": "horizontal"},
        True,
    ),
    (
        "RandomResize",
        layers.RandomResize,
        {"heights": [2]},
        False,  # tf.image.resize
    ),
    (
        "RandomRotate",
        layers.RandomRotate,
        {"factor": 10},
        False,  # tf.raw_ops.ImageProjectiveTransformV3
    ),
    (
        "RandomZoomAndCrop",
        layers.RandomZoomAndCrop,
        {"height": 2, "width": 2, "scale_factor": (0.8, 1.25)},
        False,  # tf.image.resize
    ),
    (
        "ChannelShuffle",
        layers.ChannelShuffle,
        {"groups": 3},
        True,
    ),
    (
        "RandomBlur",
        layers.RandomBlur,
        {"factor": (3, 7)},
        False,  # tf.map_fn
    ),
    (
        "RandomChannelShift",
        layers.RandomChannelShift,
        {"value_range": (0, 255), "factor": 0.1},
        True,
    ),
    (
        "RandomCLAHE",
        layers.RandomCLAHE,
        {"value_range": (0, 255), "factor": (2, 10), "tile_grid_size": (4, 4)},
        False,
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
        True,
    ),
    (
        "RandomGamma",
        layers.RandomGamma,
        {"value_range": (0, 255), "factor": 0.1},
        True,
    ),
    (
        "RandomGaussianBlur",
        layers.RandomGaussianBlur,
        {"kernel_size": 3, "factor": 2.0},
        True,
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
        True,
    ),
    (
        "RandomJpegQuality",
        layers.RandomJpegQuality,
        {
            "value_range": (0, 255),
            "factor": (75, 100),
        },
        False,  # tf.image.adjust_jpeg_quality
    ),
    (
        "RandomPosterize",
        layers.RandomPosterize,
        {"value_range": (0, 255), "factor": (5, 8)},
        True,
    ),
    (
        "RandomSharpness",
        layers.RandomSharpness,
        {"value_range": (0, 255), "factor": 0.1},
        True,
    ),
    (
        "RandomSolarize",
        layers.RandomSolarize,
        {
            "value_range": (0, 255),
            "threshold_factor": 10,
            "addition_factor": 10,
        },
        True,
    ),
    (
        "CutMix",
        layers.CutMix,
        {"alpha": 1.0},
        False,
    ),
    (
        "MixUp",
        layers.MixUp,
        {},
        False,
    ),
    (
        "Mosaic",
        layers.Mosaic,
        {
            "height": 100,
            "width": 100,
        },
        False,  # tf.map_fn
    ),
    (
        "RandomChannelDropout",
        layers.RandomChannelDropout,
        {},
        True,
    ),
    (
        "RandomCutout",
        layers.RandomCutout,
        {"height_factor": 0.3, "width_factor": 0.3},
        True,
    ),
    (
        "RandomErase",
        layers.RandomErase,
        {"area_factor": (0.02, 0.4), "aspect_ratio_factor": (0.3, 1.0 / 0.3)},
        True,
    ),
    (
        "RandomGridMask",
        layers.RandomGridMask,
        {
            "size_factor": (0.5, 1.0),
            "ratio_factor": (0.6, 0.6),
            "rotation_factor": (-10, 10),
        },
        False,  # tf.raw_ops.ImageProjectiveTransformV3
    ),
    (
        "RandomApply",
        layers.RandomApply,
        {"layer": layers.RandomChannelDropout()},
        True,  # depends on the `layer`
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
        True,  # depends on the `layers`
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
        False,  # tf.random.state_less.shuffle
    ),
    (
        "CenterCrop",
        layers.CenterCrop,
        {"height": 2, "width": 2},
        True,
    ),
    (
        "PadIfNeeded",
        layers.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
        True,
    ),
    (
        "Resize",
        layers.Resize,
        {"height": 2, "width": 2},
        True,
    ),
    (
        "AutoContrast",
        layers.AutoContrast,
        {"value_range": (0, 255)},
        True,
    ),
    (
        "Equalize",
        layers.Equalize,
        {"value_range": (0, 255)},
        False,  # tf.histogram_fixed_width
    ),
    (
        "Grayscale",
        layers.Grayscale,
        {"output_channels": 3},
        True,
    ),
    (
        "Identity",
        layers.Identity,
        {},
        True,
    ),
    (
        "Invert",
        layers.Invert,
        {"value_range": (0, 255)},
        True,
    ),
    (
        "Normalize",
        layers.Normalize,
        {"value_range": (0, 255)},
        True,
    ),
    (
        "Rescale",
        layers.Rescale,
        {"scale": 1.0 / 255.0},
        True,
    ),
]

MUST_RUN_WITH_BOUNDING_BOXES = [
    (
        "SanitizeBoundingBox",
        layers.SanitizeBoundingBox,
        {"min_size": 10},
        False,  # bounding_box.to_dense
    ),
]


class GraphAndXLAModeTest(tf.test.TestCase, parameterized.TestCase):
    def test_all_2d_aug_layers_are_included(self):
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
        bbox_names = set(item[0] for item in MUST_RUN_WITH_BOUNDING_BOXES)
        all_test_names = general_names.union(bbox_names)

        for name in all_2d_aug_layer_names:
            self.assertIn(name, all_test_names, msg=f"{name} not found")

    @parameterized.named_parameters(*GENERAL_TESTS)
    @pytest.mark.large
    def test_run_in_graph_mode(self, layer_cls, args, is_xla_compatible):
        images = tf.random.uniform(shape=(2, 8, 8, 3)) * 255.0
        labels = tf.random.uniform(shape=(2, 1)) * 10.0
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 1], [2]], dtype=tf.float32),
        }
        try:
            layer = layer_cls(**args, bounding_box_format="xyxy")
            has_bounding_boxes = True
        except TypeError:
            layer = layer_cls(**args)
            has_bounding_boxes = False

        @tf.function
        def fn(inputs):
            layer(inputs)

        if has_bounding_boxes:
            fn({IMAGES: images, LABELS: labels, BOUNDING_BOXES: bounding_boxes})
        else:
            fn({IMAGES: images, LABELS: labels})

    @parameterized.named_parameters(*GENERAL_TESTS)
    @pytest.mark.large
    def test_run_in_xla_mode(self, layer_cls, args, is_xla_compatible):
        if is_xla_compatible is False:
            return
        images = tf.random.uniform(shape=(1, 4, 4, 3)) * 255.0
        labels = tf.random.uniform(shape=(1, 1)) * 10.0
        layer = layer_cls(**args)

        @tf.function(jit_compile=True)
        def fn(inputs):
            layer(inputs)

        fn({IMAGES: images, LABELS: labels})

    @parameterized.named_parameters(*MUST_RUN_WITH_BOUNDING_BOXES)
    @pytest.mark.large
    def test_run_in_graph_mode_bbox(self, layer_cls, args, is_xla_compatible):
        images = tf.random.uniform(shape=(2, 4, 4, 3)) * 255.0
        labels = tf.random.uniform(shape=(2, 1)) * 10.0
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[0, 0, 1, 1], [0, 0, 4, 4]],
                    [[0, 0, 2, 3]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 1], [2]], dtype=tf.float32),
        }
        layer = layer_cls(**args, bounding_box_format="xyxy")

        @tf.function
        def fn(inputs):
            layer(inputs)

        fn({IMAGES: images, LABELS: labels, BOUNDING_BOXES: bounding_boxes})

import inspect

import tensorflow as tf
from absl.testing import parameterized
from keras_cv import bounding_box

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
#       is_bbox_compatible,
#   )
# all configurations should be expanded for readability
GENERAL_TESTS = [
    (
        "RandAugment",
        layers.RandAugment,
        {"value_range": (0, 255), "seed": 2023},
        True,
    ),
    (
        "TrivialAugmentWide",
        layers.TrivialAugmentWide,
        {"value_range": (0, 255)},
        True,
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
        True,
    ),
    (
        "RandomCrop",
        layers.RandomCrop,
        {"height": 2, "width": 2},
        True,
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
        True,
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
        True,
    ),
    (
        "RandomRotate",
        layers.RandomRotate,
        {"factor": 10},
        True,
    ),
    (
        "RandomZoomAndCrop",
        layers.RandomZoomAndCrop,
        {"height": 2, "width": 2, "scale_factor": (0.8, 1.25)},
        True,
    ),
    (
        "MixUp",
        layers.MixUp,
        {},
        True,
    ),
    (
        "Mosaic",
        layers.Mosaic,
        {
            "height": 2,
            "width": 2,
        },
        True,
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
        True,
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
        {"value_range": (0, 255)},
        True,
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
        True,
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
        "RandomGridMask",
        layers.RandomGridMask,
        {
            "size_factor": (0.5, 1.0),
            "ratio_factor": (0.6, 0.6),
            "rotation_factor": (-10, 10),
        },
        True,
    ),
    (
        "RandomApply",
        layers.RandomApply,
        {"layer": layers.RandomChannelDropout()},
        True,
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
        True,
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
        True,
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
        True,
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
    (
        "SanitizeBoundingBox",
        layers.SanitizeBoundingBox,
        {"min_size": 10},
        True,
    ),
]

NO_BOUNDING_BOXES_SUPPORT = [
    (
        "AugMix",
        layers.AugMix,
        {"value_range": (0, 255)},
        False,
    ),
    (
        "CutMix",
        layers.CutMix,
        {"alpha": 1.0},
        False,
    ),
    (
        "RandomErase",
        layers.RandomErase,
        {"area_factor": (0.02, 0.4), "aspect_ratio_factor": (0.3, 1.0 / 0.3)},
        False,
    ),
]


class WithBoundingBoxesTest(tf.test.TestCase, parameterized.TestCase):
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

        cosistent_names = set(item[0] for item in GENERAL_TESTS)
        no_support_names = set(item[0] for item in NO_BOUNDING_BOXES_SUPPORT)
        all_test_names = cosistent_names.union(no_support_names)

        for name in all_2d_aug_layer_names:
            self.assertIn(name, all_test_names, msg=f"{name} not found")

    @parameterized.named_parameters(*GENERAL_TESTS)
    def test_ragged_bounding_boxes(self, layer_cls, args, is_bbox_compatible):
        images = tf.random.uniform(shape=(2, 16, 16, 3)) * 255.0
        labels = tf.ones(shape=(2, 1))
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[1, 1, 2, 2], [10, 10, 15, 15]],
                    [[2, 2, 4, 4]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 1], [2]], dtype=tf.float32),
        }
        if is_bbox_compatible:
            try:
                layer = layer_cls(**args, bounding_box_format="xyxy")
            except TypeError:
                layer = layer_cls(**args)
            inputs = {
                IMAGES: images,
                LABELS: labels,
                BOUNDING_BOXES: bounding_boxes,
            }
        else:
            layer = layer_cls(**args)
            inputs = {IMAGES: images, LABELS: labels}

        outputs = layer(inputs)

        bounding_boxes = outputs[BOUNDING_BOXES]
        bounding_box.validate_format(bounding_boxes)
        self.assertTrue(isinstance(bounding_boxes["boxes"], tf.RaggedTensor))

    @parameterized.named_parameters(*GENERAL_TESTS)
    def test_dense_bounding_boxes(self, layer_cls, args, is_bbox_compatible):
        images = tf.random.uniform(shape=(2, 16, 16, 3)) * 255.0
        labels = tf.random.uniform(shape=(2, 1)) * 10.0
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[1, 1, 2, 2], [10, 10, 15, 15]],
                    [[2, 2, 4, 4]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 1], [2]], dtype=tf.float32),
        }
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        if is_bbox_compatible:
            try:
                layer = layer_cls(**args, bounding_box_format="xyxy")
            except TypeError:
                layer = layer_cls(**args)
            inputs = {
                IMAGES: images,
                LABELS: labels,
                BOUNDING_BOXES: bounding_boxes,
            }
        else:
            layer = layer_cls(**args)
            inputs = {IMAGES: images, LABELS: labels}

        outputs = layer(inputs)

        bounding_boxes = outputs[BOUNDING_BOXES]
        bounding_box.validate_format(bounding_boxes)
        self.assertTrue(isinstance(bounding_boxes["boxes"], tf.Tensor))

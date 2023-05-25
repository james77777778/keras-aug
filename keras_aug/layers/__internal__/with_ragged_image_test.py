import inspect

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
#       is_bbox_compatible,
#   )
# all configurations should be expanded for readability
CONSISTENT_OUTPUTS_LAYERS = [
    (
        "AugMix",
        layers.AugMix,
        {"value_range": (0, 255)},
        False,
    ),
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
        "RandomFlip",
        layers.RandomFlip,
        {"mode": "horizontal"},
        True,
    ),
    (
        "RandomRotate",
        layers.RandomRotate,
        {"factor": 10},
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
        "RandomErase",
        layers.RandomErase,
        {"area_factor": (0.02, 0.4), "aspect_ratio_factor": (0.3, 1.0 / 0.3)},
        False,
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
        "PadIfNeeded",
        layers.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
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

FORCE_DENSE_IMAGES_LAYERS = [
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
        "RandomResize",
        layers.RandomResize,
        {"heights": [2]},
        True,
    ),
    (
        "RandomZoomAndCrop",
        layers.RandomZoomAndCrop,
        {"height": 2, "width": 2, "scale_factor": (0.8, 1.25)},
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
        "CenterCrop",
        layers.CenterCrop,
        {"height": 2, "width": 2},
        True,
    ),
    (
        "Resize",
        layers.Resize,
        {"height": 2, "width": 2},
        True,
    ),
]

NO_RAGGED_IMAGES_SUPPORT = [
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
        True,
    ),
]


class WithRaggedImageTest(tf.test.TestCase, parameterized.TestCase):
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

        cosistent_names = set(item[0] for item in CONSISTENT_OUTPUTS_LAYERS)
        dense_names = set(item[0] for item in FORCE_DENSE_IMAGES_LAYERS)
        no_ragged_names = set(item[0] for item in NO_RAGGED_IMAGES_SUPPORT)
        all_test_names = cosistent_names.union(dense_names).union(
            no_ragged_names
        )

        for name in all_2d_aug_layer_names:
            self.assertIn(name, all_test_names, msg=f"{name} not found")

    @parameterized.named_parameters(*CONSISTENT_OUTPUTS_LAYERS)
    def test_consistent_images(self, layer_cls, args, is_bbox_compatible):
        images = tf.ragged.stack([tf.ones((5, 5, 3)), tf.ones((8, 8, 3))])
        labels = tf.ragged.stack([tf.ones((1,)), tf.ones((1,))])
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

        self.assertTrue(isinstance(outputs[IMAGES], tf.RaggedTensor))

    @parameterized.named_parameters(*FORCE_DENSE_IMAGES_LAYERS)
    def test_force_dense_images(self, layer_cls, args, is_bbox_compatible):
        images = tf.ragged.stack([tf.ones((5, 5, 3)), tf.ones((8, 8, 3))])
        labels = tf.ragged.stack([tf.ones((1,)), tf.ones((1,))])
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

        self.assertTrue(isinstance(outputs[IMAGES], tf.Tensor))

    @parameterized.named_parameters(*NO_RAGGED_IMAGES_SUPPORT)
    def test_no_ragged_images(self, layer_cls, args, is_bbox_compatible):
        images = tf.ragged.stack([tf.ones((5, 5, 3)), tf.ones((8, 8, 3))])
        labels = tf.ragged.stack([tf.ones((1,)), tf.ones((1,))])
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

        with self.assertRaises(ValueError):
            layer(inputs)

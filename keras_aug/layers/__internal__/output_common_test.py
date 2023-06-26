import copy
import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers
from keras_aug.layers import augmentation
from keras_aug.layers import preprocessing
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

SEED = 2025
#   (
#       name,
#       layer_cls,
#       args,
#       is_bbox_compatible,
#   )
# all configurations should be expanded for readability
GENERAL_TESTS = [
    (
        "AugMix",
        layers.AugMix,
        {"value_range": (0, 255)},
        False,
    ),
    (
        "RandAugment",
        layers.RandAugment,
        {"value_range": (0, 255)},
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
        {"mode": "horizontal_and_vertical"},
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
        {
            "value_range": (0, 255),
            "factor": (1, 100),
            "tile_grid_size": (4, 4),
            "seed": 2024,  # manually set
        },
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
    (
        "Mosaic",
        layers.Mosaic,
        {
            "height": 100,
            "width": 100,
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
        {
            "layer": layers.RandomColorJitter(
                value_range=(0, 255), brightness_factor=(0.5, 1.5), seed=SEED
            ),
            "seed": 2024,
        },
        True,
    ),
    (
        "RandomChoice",
        layers.RandomChoice,
        {
            "layers": [
                layers.RandomColorJitter(
                    value_range=(0, 255), brightness_factor=(0.5, 0.5)
                ),
                layers.RandomColorJitter(
                    value_range=(0, 255), brightness_factor=(1.5, 1.5)
                ),
            ],
            "seed": 2024,
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

NO_PRESERVED_SHAPE = [
    layers.RandomCrop,
    layers.RandomCropAndResize,
    layers.RandomResize,
    layers.RandomZoomAndCrop,
    layers.Mosaic,
    layers.RepeatedAugment,
    layers.CenterCrop,
    layers.Resize,
]

NO_UINT8 = [
    layers.AugMix,  # alpha
    layers.RandAugment,  # stateless_random_uniform
    layers.TrivialAugmentWide,  # stateless_random_uniform
    layers.RandomSharpness,  # stateless_random_uniform
    layers.RandomSolarize,  # stateless_random_uniform
    layers.CutMix,  # tf.convert_to_tensor
    layers.MixUp,  # tf.convert_to_tensor
    layers.RandomCutout,  # tf.where with -1 (invalid bbox)
    layers.RandomHSV,  # stateless_random_uniform
    layers.RandomChannelShift,  # stateless_random_uniform
    layers.RandomColorJitter,  # stateless_random_uniform
    layers.RandomGamma,  # stateless_random_uniform
    layers.RandomGaussianBlur,  # tf.nn.depthwise_conv2d
    layers.RandomJpegQuality,  # preprocessing_utils.transform_value_range
    layers.AutoContrast,  # tf.convert_to_tensor
    layers.Grayscale,  # tf.mul
    layers.Normalize,  # mean, std
]

SKIP_DTYPE = [
    # it is hard to change dtype in runtime
    layers.RandomApply,
    layers.RepeatedAugment,
]

ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS = [
    layers.RandomResize,  # same size in the batch
    layers.CutMix,  # cannot test CutMix with same images
    layers.MixUp,  # cannot test MixUp with same images
    layers.CenterCrop,
    layers.PadIfNeeded,
    layers.Resize,
    layers.AutoContrast,
    layers.Equalize,
    layers.Grayscale,
    layers.Identity,
    layers.Invert,
    layers.Normalize,
    layers.Rescale,
    layers.SanitizeBoundingBox,
]


class OutputCommonTest(tf.test.TestCase, parameterized.TestCase):
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
    def test_preserves_output_shape(self, layer_cls, args, is_bbox_compatible):
        images = tf.random.uniform(shape=(2, 16, 16, 3), seed=SEED) * 255.0
        labels = tf.random.uniform(shape=(2, 1), seed=SEED) * 10.0
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[0, 0, 2, 2], [0, 0, 16, 16]],
                    [[2, 5, 1, 4]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [
                    [0, 1],
                    [2],
                ],
                dtype=tf.float32,
            ),
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

        if layer_cls not in NO_PRESERVED_SHAPE:
            self.assertEqual(images.shape, outputs[IMAGES].shape)
        else:
            self.assertNotEqual(images.shape, outputs[IMAGES].shape)

    @parameterized.named_parameters(*GENERAL_TESTS)
    def test_layer_dtypes(self, layer_cls, args, is_bbox_compatible):
        if layer_cls in SKIP_DTYPE:
            return
        images = tf.random.uniform(shape=(2, 16, 16, 3), seed=SEED) * 255.0
        labels = tf.random.uniform(shape=(2, 1), seed=SEED) * 10.0
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[0, 0, 2, 2], [0, 0, 16, 16]],
                    [[2, 5, 1, 4]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [
                    [0, 1],
                    [2],
                ],
                dtype=tf.float32,
            ),
        }
        copied_args = copy.deepcopy(args)
        if is_bbox_compatible:
            try:
                layer_cls(**copied_args, bounding_box_format="xyxy")
                copied_args["bounding_box_format"] = "xyxy"
            except TypeError:
                pass
            inputs = {
                IMAGES: images,
                LABELS: labels,
                BOUNDING_BOXES: bounding_boxes,
            }
        else:
            inputs = {IMAGES: images, LABELS: labels}

        # float64
        layer = layer_cls(**copied_args, dtype=tf.float64)
        results = layer(inputs)
        self.assertDTypeEqual(results[IMAGES], tf.float64)

        # float32
        layer = layer_cls(**copied_args)
        results = layer(inputs)
        self.assertDTypeEqual(results[IMAGES], tf.float32)

        # float16
        layer = layer_cls(**copied_args, dtype=tf.float16)
        results = layer(inputs)
        self.assertDTypeEqual(results[IMAGES], tf.float16)

        # bfloat16
        layer = layer_cls(**copied_args, dtype=tf.bfloat16)
        results = layer(inputs)
        self.assertDTypeEqual(results[IMAGES], tf.bfloat16)

        # uint8
        if layer_cls not in NO_UINT8:
            layer = layer_cls(**copied_args, dtype=tf.uint8)
            results = layer(inputs)
            self.assertDTypeEqual(results[IMAGES], tf.uint8)
        else:
            with self.assertRaises(
                (TypeError, ValueError, tf.errors.InvalidArgumentError)
            ):
                layer = layer_cls(**copied_args, dtype=tf.uint8)
                layer(inputs)

    @parameterized.named_parameters(*GENERAL_TESTS)
    def test_independence_on_batched(self, layer_cls, args, is_bbox_compatible):
        image = tf.random.uniform((16, 16, 3), seed=SEED) * 255.0
        label = tf.random.uniform((1,), seed=SEED) * 255.0
        batched_images = tf.stack((image, image, image, image, image), axis=0)
        batched_labels = tf.stack((label, label, label, label, label), axis=0)
        batched_bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=tf.float32
            ),
        }
        copied_args = copy.deepcopy(args)
        if is_bbox_compatible:
            try:
                layer_cls(**copied_args, bounding_box_format="xyxy")
                copied_args["bounding_box_format"] = "xyxy"
            except TypeError:
                pass
            inputs = {
                IMAGES: batched_images,
                LABELS: batched_labels,
                BOUNDING_BOXES: batched_bounding_boxes,
            }
        else:
            inputs = {IMAGES: batched_images, LABELS: batched_labels}
        try:
            layer_cls(**copied_args, seed=SEED)
            copied_args["seed"] = SEED
        except TypeError:
            pass
        layer = layer_cls(**copied_args)

        results = layer(inputs)

        if layer_cls not in ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS:
            self.assertNotAllClose(results[IMAGES][0], results[IMAGES][1])
        else:
            self.assertAllClose(results[IMAGES][0], results[IMAGES][1])

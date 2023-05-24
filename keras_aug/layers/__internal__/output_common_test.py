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
    (
        "RandomFlip",
        layers.RandomFlip,
        {"mode": "horizontal_and_vertical", "seed": 1},
    ),
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
        {"value_range": (0, 255), "factor": (1, 100), "tile_grid_size": (2, 2)},
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
        {"layer": layers.RandomChannelDropout(seed=SEED)},
    ),
    (
        "RandomChoice",
        layers.RandomChoice,
        {
            "layers": [
                layers.RandomChannelDropout(seed=1),
                layers.RandomChannelDropout(seed=5),
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
    (
        "SanitizeBoundingBox",
        layers.SanitizeBoundingBox,
        {"min_size": 10},
    ),
    (
        "Resize",
        layers.Resize,
        {"height": 2, "width": 2},
    ),
]

NO_PRESERVED_SHAPE_LAYERS = [
    layers.RandomCrop,
    layers.RandomCropAndResize,
    layers.RandomResize,
    layers.RandomZoomAndCrop,
    layers.Mosaic,
    layers.RepeatedAugment,
    layers.CenterCrop,
    layers.Resize,
]

NO_BFLOAT16_DTYPE_LAYERS = [
    layers.RandAugment,
    layers.RandomAffine,
    layers.RandomCrop,
    layers.RandomCropAndResize,
    layers.RandomRotate,
]

SKIP_DTYPE_LAYERS = [
    # hard to test the policy of RandAugment
    layers.RandAugment,
    # it is impossible to change dtype in runtime for RandomApply
    layers.RandomApply,
    layers.RepeatedAugment,
]

ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS = [
    layers.CenterCrop,
    layers.PadIfNeeded,
    layers.Resize,
    layers.AutoContrast,
    layers.Equalize,
    layers.Grayscale,
    layers.Invert,
    layers.Normalize,
    layers.Rescale,
    layers.CutMix,  # cannot test CutMix with same images
    layers.MixUp,  # cannot test MixUp with same images
    layers.Identity,
    layers.RandomApply,
    layers.SanitizeBoundingBox,
]


class OutputCommonTest(tf.test.TestCase, parameterized.TestCase):
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
    def test_preserves_output_shape(self, layer_cls, args):
        images = tf.random.uniform(shape=(2, 16, 16, 3), seed=SEED) * 255.0
        labels = tf.random.uniform(shape=(2, 1), seed=SEED) * 10.0
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
            # try bounding_box_format
            layer_cls(**args, bounding_box_format="xyxy")
            args["bounding_box_format"] = "xyxy"
            inputs = {
                IMAGES: images,
                LABELS: labels,
                BOUNDING_BOXES: bounding_boxes,
            }
        except (TypeError, ValueError):
            inputs = {IMAGES: images, LABELS: labels}
        try:
            layer_cls(**args, seed=SEED)
            args["seed"] = SEED
        except TypeError:
            pass
        layer = layer_cls(**args)

        outputs = layer(inputs)

        if layer_cls not in NO_PRESERVED_SHAPE_LAYERS:
            self.assertEqual(images.shape, outputs[IMAGES].shape)
        else:
            self.assertNotEqual(images.shape, outputs[IMAGES].shape)

        # clean up
        args.pop("bounding_box_format", None)
        args.pop("seed")

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_layer_dtypes(self, layer_cls, args):
        if layer_cls in SKIP_DTYPE_LAYERS:
            return
        images = tf.cast(
            tf.random.uniform(shape=(2, 16, 16, 3), seed=SEED) * 255.0,
            dtype=tf.float64,
        )
        labels = tf.cast(
            tf.random.uniform(shape=(2, 1), seed=SEED) * 10.0, dtype=tf.float64
        )
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
            # try bounding_box_format
            layer_cls(**args, bounding_box_format="xyxy")
            args["bounding_box_format"] = "xyxy"
            inputs = {
                IMAGES: images,
                LABELS: labels,
                BOUNDING_BOXES: bounding_boxes,
            }
        except (TypeError, ValueError):
            inputs = {IMAGES: images, LABELS: labels}
        try:
            layer_cls(**args, seed=SEED)
            args["seed"] = SEED
        except TypeError:
            pass

        # float32
        layer = layer_cls(**args)
        results = layer(inputs)
        self.assertAllEqual(results[IMAGES].dtype, "float32")

        # float16
        layer = layer_cls(**args, dtype="float16")
        results = layer(inputs)
        self.assertAllEqual(results[IMAGES].dtype, "float16")

        # bfloat16
        if layer_cls not in NO_BFLOAT16_DTYPE_LAYERS:
            layer = layer_cls(**args, dtype="bfloat16")
            results = layer(inputs)
            self.assertAllEqual(results[IMAGES].dtype, "bfloat16")
        else:
            with self.assertRaises(
                (TypeError, ValueError, tf.errors.InvalidArgumentError)
            ):
                layer = layer_cls(**args, dtype="bfloat16")
                results = layer(inputs)

        # clean up
        args.pop("bounding_box_format", None)
        args.pop("seed")

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_independence_on_batched_images(self, layer_cls, args):
        image = tf.random.uniform((16, 16, 3), seed=SEED) * 255.0
        label = tf.random.uniform((1,), seed=SEED) * 255.0
        batched_images = tf.stack((image, image, image, image, image), axis=0)
        batched_labels = tf.stack((label, label, label, label, label), axis=0)
        batched_bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[200, 200, 400, 400]],
                    [[200, 200, 400, 400]],
                    [[200, 200, 400, 400]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [[0, 1], [2], [2], [2], [2]], dtype=tf.float32
            ),
        }
        try:
            # try bounding_box_format
            layer_cls(**args, bounding_box_format="xyxy")
            args["bounding_box_format"] = "xyxy"
            inputs = {
                IMAGES: batched_images,
                LABELS: batched_labels,
                BOUNDING_BOXES: batched_bounding_boxes,
            }
        except (TypeError, ValueError):
            inputs = {IMAGES: batched_images, LABELS: batched_labels}
        try:
            layer_cls(**args, seed=SEED)
            args["seed"] = SEED
        except TypeError:
            pass
        layer = layer_cls(**args)

        results = layer(inputs)

        if layer_cls not in ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS:
            self.assertNotAllClose(results[IMAGES][0], results[IMAGES][1])
        else:
            self.assertAllClose(results[IMAGES][0], results[IMAGES][1])

        # clean up
        args.pop("bounding_box_format", None)
        args.pop("seed")

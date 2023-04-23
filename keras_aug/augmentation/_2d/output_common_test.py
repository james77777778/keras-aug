import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug.augmentation import _2d as augmentation
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS

SEED = 2025
TEST_CONFIGURATIONS = [
    (
        "CenterCrop",
        augmentation.CenterCrop,
        {"height": 2, "width": 2},
    ),
    (
        "PadIfNeeded",
        augmentation.PadIfNeeded,
        {"min_height": 20, "min_width": 20},
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
        "RandomCLAHE",
        augmentation.RandomCLAHE,
        {"value_range": (0, 255), "factor": (2, 20), "tile_grid_size": (4, 4)},
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
        {"value_range": (0, 255), "factor": (75, 100)},
    ),
    (
        "Rescaling",
        augmentation.Rescaling,
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
        {"height": 20, "width": 20},
    ),
    (
        "ChannelDropout",
        augmentation.ChannelDropout,
        {},
    ),
    (
        "RandomApply",
        augmentation.RandomApply,
        {"layer": augmentation.ChannelDropout()},
    ),
]


NO_PRESERVED_SHAPE_LAYERS = [
    augmentation.CenterCrop,
    augmentation.PadIfNeeded,
    augmentation.MosaicYOLOV8,
    augmentation.RandomCropAndResize,
    augmentation.ResizeAndPad,
    augmentation.ResizeByLongestSide,
    augmentation.ResizeBySmallestSide,
]

NO_BFLOAT16_DTYPE_LAYERS = [
    augmentation.RandomAffine,
    augmentation.RandomCropAndResize,
]

NO_UINT8_DTYPE_LAYERS = [
    augmentation.Normalize,
    augmentation.RandomBrightnessContrast,
    augmentation.RandomColorJitter,
    augmentation.RandomGamma,
    augmentation.RandomHSV,
    augmentation.RandomJpegQuality,
    augmentation.MixUp,
    augmentation.MosaicYOLOV8,
]

SKIP_DTYPE_LAYERS = [
    # it is impossible to change dtype in runtime for RandomApply
    augmentation.RandomApply,
]

ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS = [
    augmentation.CenterCrop,
    augmentation.PadIfNeeded,
    augmentation.ResizeAndPad,
    augmentation.ResizeByLongestSide,
    augmentation.ResizeBySmallestSide,
    augmentation.Normalize,
    augmentation.Rescaling,
    augmentation.MixUp,
    augmentation.RandomApply,
]


class OutputCommonTest(tf.test.TestCase, parameterized.TestCase):
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
    def test_preserves_output_shape(self, layer_cls, args):
        images = tf.random.uniform(shape=(4, 16, 16, 3), seed=SEED) * 255.0
        labels = tf.random.uniform(shape=(4, 1), seed=SEED) * 10.0
        layer = layer_cls(**args, seed=SEED)

        outputs = layer({IMAGES: images, LABELS: labels})

        if layer_cls not in NO_PRESERVED_SHAPE_LAYERS:
            self.assertEqual(images.shape, outputs[IMAGES].shape)
            self.assertNotAllClose(images, outputs[IMAGES])
        else:
            self.assertNotEqual(images.shape, outputs[IMAGES].shape)

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_uint8_input(self, layer_cls, args):
        images = tf.cast(
            tf.random.uniform(shape=(4, 16, 16, 3), seed=SEED) * 255.0,
            dtype=tf.uint8,
        )
        labels = tf.cast(
            tf.random.uniform(shape=(4, 1), seed=SEED) * 10.0, dtype=tf.uint8
        )
        layer = layer_cls(**args, seed=SEED)

        output = layer({IMAGES: images, LABELS: labels})

        self.assertNotAllClose(images, output[IMAGES])

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_layer_dtypes(self, layer_cls, args):
        if layer_cls in SKIP_DTYPE_LAYERS:
            return
        images = tf.cast(
            tf.random.uniform(shape=(4, 16, 16, 3), seed=SEED) * 255.0,
            dtype=tf.float64,
        )
        labels = tf.cast(
            tf.random.uniform(shape=(4, 1), seed=SEED) * 10.0, dtype=tf.float64
        )

        # float32
        layer = layer_cls(**args, seed=SEED)
        results = layer({IMAGES: images, LABELS: labels})
        self.assertAllEqual(results[IMAGES].dtype, "float32")

        # float16
        layer = layer_cls(**args, seed=SEED, dtype="float16")
        results = layer({IMAGES: images, LABELS: labels})
        self.assertAllEqual(results[IMAGES].dtype, "float16")

        # bfloat16
        if layer_cls not in NO_BFLOAT16_DTYPE_LAYERS:
            layer = layer_cls(**args, seed=SEED, dtype="bfloat16")
            results = layer({IMAGES: images, LABELS: labels})
            self.assertAllEqual(results[IMAGES].dtype, "bfloat16")
        else:
            with self.assertRaises(
                (TypeError, ValueError, tf.errors.InvalidArgumentError)
            ):
                layer = layer_cls(**args, seed=SEED, dtype="bfloat16")
                results = layer({IMAGES: images, LABELS: labels})

        # uint8
        if layer_cls not in NO_UINT8_DTYPE_LAYERS:
            layer = layer_cls(**args, seed=SEED, dtype="uint8")
            results = layer({IMAGES: images, LABELS: labels})
            self.assertAllEqual(results[IMAGES].dtype, "uint8")
        else:
            with self.assertRaises(
                (TypeError, ValueError, tf.errors.InvalidArgumentError)
            ):
                layer = layer_cls(**args, seed=SEED, dtype="uint8")
                results = layer({IMAGES: images, LABELS: labels})

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_independence_on_batched_images(self, layer_cls, args):
        image = tf.random.uniform((16, 16, 3), seed=SEED) * 255.0
        label = tf.random.uniform((1,), seed=SEED) * 255.0
        batched_images = tf.stack((image, image, image, image, image), axis=0)
        batched_labels = tf.stack((label, label, label, label, label), axis=0)
        layer = layer_cls(**args, seed=SEED)

        results = layer({IMAGES: batched_images, LABELS: batched_labels})

        if layer_cls not in ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS:
            self.assertNotAllClose(results[IMAGES][0], results[IMAGES][1])
        else:
            self.assertAllClose(results[IMAGES][0], results[IMAGES][1])

import inspect

import tensorflow as tf
from absl.testing import parameterized

from keras_aug.augmentations import _2d as augmentations
from keras_aug.utils.augmentation_utils import IMAGES
from keras_aug.utils.augmentation_utils import LABELS

SEED = 2025
TEST_CONFIGURATIONS = [
    (
        "CenterCrop",
        augmentations.CenterCrop,
        {"height": 2, "width": 2},
    ),
    (
        "PadIfNeeded",
        augmentations.PadIfNeeded,
        {"min_height": 2, "min_width": 2},
    ),
    (
        "RandomAffine",
        augmentations.RandomAffine,
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
        augmentations.RandomCropAndResize,
        {
            "height": 2,
            "width": 2,
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
    ),
    (
        "ResizeAndPad",
        augmentations.ResizeAndPad,
        {"height": 2, "width": 2},
    ),
    (
        "ResizeByLongestSide",
        augmentations.ResizeByLongestSide,
        {"max_size": [2]},
    ),
    (
        "ResizeBySmallestSide",
        augmentations.ResizeBySmallestSide,
        {"min_size": [2]},
    ),
    (
        "CLAHE",
        augmentations.CLAHE,
        {"value_range": (0, 255), "factor": (2, 10), "tile_grid_size": (4, 4)},
    ),
    ("Normalize", augmentations.Normalize, {"value_range": (0, 255)}),
    ("RandomBlur", augmentations.RandomBlur, {"factor": (3, 7)}),
    (
        "RandomBrightnessContrast",
        augmentations.RandomBrightnessContrast,
        {
            "value_range": (0, 255),
            "brightness_factor": 0.1,
            "contrast_factor": 0.1,
        },
    ),
    (
        "RandomColorJitter",
        augmentations.RandomColorJitter,
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
        augmentations.RandomGamma,
        {"value_range": (0, 255), "factor": 0.1},
    ),
    (
        "RandomHSV",
        augmentations.RandomHSV,
        {
            "value_range": (0, 255),
            "hue_factor": 0.1,
            "saturation_factor": 0.1,
            "value_factor": 0.1,
        },
    ),
    (
        "RandomJpegQuality",
        augmentations.RandomJpegQuality,
        {
            "value_range": (0, 255),
            "factor": (75, 100),
        },
    ),
    (
        "MosaicYOLOV8",
        augmentations.MosaicYOLOV8,
        {
            "height": 100,
            "width": 100,
        },
    ),
    (
        "ChannelDropout",
        augmentations.ChannelDropout,
        {},
    ),
]


NO_PRESERVED_SHAPE_LAYERS = [
    augmentations.CenterCrop,
    augmentations.PadIfNeeded,
    augmentations.MosaicYOLOV8,
    augmentations.RandomCropAndResize,
    augmentations.ResizeAndPad,
    augmentations.ResizeByLongestSide,
    augmentations.ResizeBySmallestSide,
]

ALWAYS_SAME_OUTPUT_LAYERS = [augmentations.PadIfNeeded]

NO_BFLOAT16_DTYPE_LAYERS = [
    augmentations.RandomAffine,
    augmentations.RandomCropAndResize,
]

NO_UINT8_DTYPE_LAYERS = [
    augmentations.Normalize,
    augmentations.RandomBrightnessContrast,
    augmentations.RandomColorJitter,
    augmentations.RandomGamma,
    augmentations.RandomHSV,
    augmentations.RandomJpegQuality,
    augmentations.MosaicYOLOV8,
]

ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS = [
    augmentations.CenterCrop,
    augmentations.PadIfNeeded,
    augmentations.ResizeAndPad,
    augmentations.ResizeByLongestSide,
    augmentations.ResizeBySmallestSide,
    augmentations.Normalize,
]


class OutputCommonTest(tf.test.TestCase, parameterized.TestCase):
    def test_all_2d_aug_layers_included(self):
        base_cls = augmentations.VectorizedBaseRandomLayer
        all_2d_aug_layers = inspect.getmembers(
            augmentations,
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
        if layer_cls in NO_PRESERVED_SHAPE_LAYERS:
            return
        image = tf.random.uniform(shape=(4, 32, 32, 3), seed=SEED) * 255.0
        layer = layer_cls(**args, seed=SEED)

        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_uint8_input(self, layer_cls, args):
        if layer_cls in ALWAYS_SAME_OUTPUT_LAYERS:
            return
        images = tf.cast(
            tf.random.uniform(shape=(4, 32, 32, 3), seed=SEED) * 255.0,
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
        images = tf.cast(
            tf.random.uniform(shape=(4, 32, 32, 3), seed=SEED) * 255.0,
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

        # uint8
        if layer_cls not in NO_UINT8_DTYPE_LAYERS:
            layer = layer_cls(**args, seed=SEED, dtype="uint8")
            results = layer({IMAGES: images, LABELS: labels})
            self.assertAllEqual(results[IMAGES].dtype, "uint8")

    @parameterized.named_parameters(*TEST_CONFIGURATIONS)
    def test_independence_on_batched_images(self, layer_cls, args):
        if layer_cls in ALWAYS_SAME_OUTPUT_WITHIN_BATCH_LAYERS:
            return
        image = tf.random.uniform((100, 100, 3), seed=SEED) * 255.0
        label = tf.random.uniform((1,), seed=SEED) * 255.0
        batched_images = tf.stack((image, image, image, image), axis=0)
        batched_labels = tf.stack((label, label, label, label), axis=0)
        layer = layer_cls(**args, seed=SEED)

        results = layer({IMAGES: batched_images, LABELS: batched_labels})

        self.assertNotAllClose(results[IMAGES][0], results[IMAGES][1])

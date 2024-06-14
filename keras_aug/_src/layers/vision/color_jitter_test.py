import keras
import numpy as np
from keras import backend
from keras import ops
from keras.src import testing

from keras_aug._src.layers.vision.color_jitter import ColorJitter


class FixedColorJitter(ColorJitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set to non-None
        self.brightness = (1.1, 1.1)
        self.contrast = (0.9, 0.9)
        self.saturation = (1.1, 1.1)
        self.hue = (-0.1, -0.1)

    def get_params(self, batch_size, images=None, **kwargs):
        return dict(
            fn_idx=(0, 1, 2, 3),
            brightness_factor=ops.ones([batch_size, 1, 1, 1]) + 0.1,
            contrast_factor=ops.ones([batch_size, 1, 1, 1]) - 0.1,
            saturation_factor=ops.ones([batch_size, 1, 1, 1]) + 0.1,
            hue_factor=ops.zeros([batch_size, 1, 1, 1]) - 0.1,
        )


class FixedFnIndexColorJitter(ColorJitter):
    def get_params(self, batch_size, images=None, **kwargs):
        transformations = super().get_params(batch_size, images, **kwargs)
        transformations["fn_idx"] = (0, 1, 2, 3)
        return transformations


class ColorJitterTest(testing.TestCase):
    regular_args = dict(
        value_range=(0, 255),
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.25,
    )

    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_correctness(self):
        import torch
        import torchvision.transforms.v2.functional as TF

        # Test channels_last
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = FixedColorJitter((0, 1))
        y = layer(x)

        ref_y = torch.tensor(np.transpose(x, [0, 3, 1, 2]))
        ref_y = TF.adjust_brightness(ref_y, 1.1)
        ref_y = TF.adjust_contrast(ref_y, 0.9)
        ref_y = TF.adjust_saturation(ref_y, 1.1)
        ref_y = TF.adjust_hue(ref_y, -0.1)
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        layer = FixedColorJitter((0, 1))
        y = layer(x)

        ref_y = torch.tensor(x)
        ref_y = TF.adjust_brightness(ref_y, 1.1)
        ref_y = TF.adjust_contrast(ref_y, 0.9)
        ref_y = TF.adjust_saturation(ref_y, 1.1)
        ref_y = TF.adjust_hue(ref_y, -0.1)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = ColorJitter(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = ColorJitter(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = ColorJitter(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = ColorJitter(**self.regular_args)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_data_format(self):
        # Test channels_last
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = ColorJitter(**self.regular_args)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 32, 32, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        layer = ColorJitter(**self.regular_args)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = FixedFnIndexColorJitter(
            (0, 255),
            brightness=(1.1, 1.1),
            contrast=(0.9, 0.9),
            saturation=(1.2, 1.2),
            hue=(-0.1, -0.1),
        )
        y = layer(x)

        layer = FixedFnIndexColorJitter.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = ColorJitter(**self.regular_args)
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (3, 32, 32, 3))

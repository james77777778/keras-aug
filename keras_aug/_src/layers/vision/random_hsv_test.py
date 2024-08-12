import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras import ops
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_hsv import RandomHSV
from keras_aug._src.testing.test_case import TestCase
from keras_aug._src.utils.test_utils import get_images


class FixedRandomHSV(RandomHSV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set to non-None
        self.hue = (0.9, 1.1)
        self.saturation = (0.9, 1.1)
        self.value = (0.9, 1.1)

    def get_params(self, batch_size, images=None, **kwargs):
        return dict(
            hue_gain=ops.ones([batch_size]) * 0.9,
            saturation_gain=ops.ones([batch_size]) * 1.1,
            value_gain=ops.ones([batch_size]) * 0.9,
        )


class RandomHSVTest(TestCase):
    regular_args = dict(hue=0.015, saturation=0.7, value=0.4)

    @parameterized.named_parameters(
        named_product(dtype=["float32", "mixed_bfloat16", "uint8"])
    )
    def test_correctness(self, dtype):
        np.random.seed(42)

        # Test channels_last
        x = get_images(dtype, "channels_last")
        layer = FixedRandomHSV(dtype=dtype)
        y = layer(x)

        # TODO: Test correctness
        self.assertEqual(y.shape, x.shape)
        self.assertDType(y, dtype)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images(dtype, "channels_first")
        layer = FixedRandomHSV(dtype=dtype)
        y = layer(x)

        # TODO: Test correctness
        self.assertEqual(y.shape, x.shape)
        self.assertDType(y, dtype)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomHSV(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomHSV(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomHSV(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomHSV(**self.regular_args)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_data_format(self):
        # Test channels_last
        x = get_images("float32", "channels_last")
        layer = RandomHSV(**self.regular_args)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 32, 32, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images("float32", "channels_first")
        layer = RandomHSV(**self.regular_args)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedRandomHSV()
        y = layer(x)

        layer = FixedRandomHSV.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomHSV(**self.regular_args)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

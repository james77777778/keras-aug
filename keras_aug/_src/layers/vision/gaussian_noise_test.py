import keras
import numpy as np
from absl.testing import parameterized
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.gaussian_noise import GaussianNoise
from keras_aug._src.utils.test_utils import get_images


class FixedGaussianNoise(GaussianNoise):
    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        compute_dtype = keras.backend.result_type(self.compute_dtype, float)
        noise = ops.numpy.ones(ops.shape(images), dtype=compute_dtype) * 0.5
        return noise


class GaussianNoiseTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        named_product(dtype=["float32", "bfloat16"])
    )
    def test_correctness(self, dtype):
        # Test channels_last
        x = get_images(dtype, "channels_last")
        layer = FixedGaussianNoise(dtype=dtype)
        y = layer(x)

        ref_y = np.clip(x + 0.5, 0, 1)
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y, atol=1e-2)

        # Test channels_first
        x = get_images(dtype, "channels_first")
        layer = FixedGaussianNoise(dtype=dtype)
        y = layer(x)

        ref_y = np.clip(x + 0.5, 0, 1)
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y, atol=1e-2)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = GaussianNoise()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = GaussianNoise()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = GaussianNoise()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedGaussianNoise()
        y = layer(x)

        layer = FixedGaussianNoise.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = GaussianNoise()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

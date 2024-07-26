import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.gaussian_blur import GaussianBlur
from keras_aug._src.utils.test_utils import get_images


class FixedGaussianBlur(GaussianBlur):
    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        compute_dtype = keras.backend.result_type(self.compute_dtype, float)
        sigma = ops.numpy.ones((), dtype=compute_dtype) * 0.1
        return sigma


class GaussianBlurTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_correctness(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        # Test channels_last
        x = get_images(dtype, "channels_last")
        layer = FixedGaussianBlur(3, dtype=dtype)
        y = layer(x)

        ref_y = TF.gaussian_blur(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), (3, 3), (0.1, 0.1)
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test channels_first
        x = get_images(dtype, "channels_first")
        layer = FixedGaussianBlur(3, dtype=dtype)
        y = layer(x)

        ref_y = TF.gaussian_blur(torch.tensor(x), (3, 3), (0.1, 0.1))
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = GaussianBlur(3)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = GaussianBlur(3)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = GaussianBlur(3)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedGaussianBlur(3)
        y = layer(x)

        layer = FixedGaussianBlur.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = GaussianBlur(3)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

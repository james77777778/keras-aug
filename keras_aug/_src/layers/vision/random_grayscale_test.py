import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_grayscale import RandomGrayscale
from keras_aug._src.utils.test_utils import get_images


class RandomGrayscaleTest(testing.TestCase, parameterized.TestCase):
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
        layer = RandomGrayscale(p=1.0, dtype=dtype)
        y = layer(x)

        ref_y = TF.rgb_to_grayscale(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), num_output_channels=3
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images(dtype, "channels_first")
        layer = RandomGrayscale(p=1.0, dtype=dtype)
        y = layer(x)

        ref_y = TF.rgb_to_grayscale(torch.tensor(x), num_output_channels=3)
        ref_y = ref_y.cpu().numpy()
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        backend.set_image_data_format("channels_last")
        x = get_images(dtype, "channels_last")
        layer = RandomGrayscale(p=0.0, dtype=dtype)
        y = layer(x)

        self.assertDType(y, dtype)
        self.assertAllClose(y, x)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomGrayscale()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomGrayscale()(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomGrayscale()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomGrayscale()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_data_format(self):
        # Test channels_last
        x = get_images("float32", "channels_last")
        layer = RandomGrayscale()
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 32, 32, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images("float32", "channels_first")
        layer = RandomGrayscale()
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = RandomGrayscale(p=1.0)
        y = layer(x)

        layer = RandomGrayscale.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomGrayscale()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

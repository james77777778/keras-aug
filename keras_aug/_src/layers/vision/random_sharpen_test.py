import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_case import tensorflow_uses_gpu
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_sharpen import RandomSharpen
from keras_aug._src.utils.test_utils import get_images


class RandomSharpenTest(testing.TestCase, parameterized.TestCase):
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
        layer = RandomSharpen(2.0, p=1.0, dtype=dtype)
        y = layer(x)

        ref_y = TF.adjust_sharpness(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), sharpness_factor=2.0
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test channels_first
        if backend.backend() == "tensorflow" and not tensorflow_uses_gpu():
            self.skipTest("Tensorflow CPU doesn't support `RandomSharpen`")
        backend.set_image_data_format("channels_first")
        x = get_images(dtype, "channels_first")
        layer = RandomSharpen(2.0, p=1.0, dtype=dtype)
        y = layer(x)

        ref_y = TF.adjust_sharpness(torch.tensor(x), sharpness_factor=2.0)
        ref_y = ref_y.cpu().numpy()
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        backend.set_image_data_format("channels_last")
        x = get_images(dtype, "channels_last")
        layer = RandomSharpen(2.0, p=0.0, dtype=dtype)
        y = layer(x)

        self.assertDType(y, dtype)
        self.assertAllClose(y, x)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomSharpen(2.0)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomSharpen(2.0)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomSharpen(2.0)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomSharpen(2.0)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_data_format(self):
        # Test channels_last
        x = get_images("float32", "channels_last")
        layer = RandomSharpen(2.0)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 32, 32, 3))

        # Test channels_first
        if backend.backend() == "tensorflow" and not tensorflow_uses_gpu():
            self.skipTest("Tensorflow CPU doesn't support `RandomSharpen`")
        backend.set_image_data_format("channels_first")
        x = get_images("float32", "channels_first")
        layer = RandomSharpen(2.0)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = RandomSharpen(2.0, p=1.0)
        y = layer(x)

        layer = RandomSharpen.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomSharpen(2.0)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

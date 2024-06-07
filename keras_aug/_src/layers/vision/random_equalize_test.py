import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing

from keras_aug._src.layers.vision.random_equalize import RandomEqualize


class RandomEqualizeTest(testing.TestCase, parameterized.TestCase):
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
        import torchvision.transforms.functional as TF

        # Test channels_last
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("uint8")
        layer = RandomEqualize(value_range=(0, 255), p=1.0)
        y = layer(x)

        ref_y = TF.equalize(torch.tensor(np.transpose(x, [0, 3, 1, 2])))
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 255, (2, 3, 32, 32)).astype("uint8")
        layer = RandomEqualize(value_range=(0, 255), p=1.0)
        y = layer(x)

        ref_y = TF.equalize(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        backend.set_image_data_format("channels_last")
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = RandomEqualize(value_range=(0, 255), p=0.0)
        y = layer(x)

        self.assertAllClose(y, x)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomEqualize(value_range=(0, 255))(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomEqualize(value_range=(0, 255))(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomEqualize(value_range=(0, 255))(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomEqualize(value_range=(0, 255))
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    @parameterized.named_parameters(
        ("float32", "float32"), ("int32", "int32"), ("int64", "int64")
    )
    def test_input_dtypes(self, dtype):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype(dtype)
        layer = RandomEqualize(value_range=(0, 255))
        y = layer(x)
        y = keras.ops.convert_to_numpy(y)
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 255))

    @parameterized.named_parameters(("0_255", 0, 255), ("0_1", 0, 1))
    def test_output_range(self, lower, upper):
        np.random.seed(42)
        x = np.random.uniform(lower, upper, (2, 4, 4, 3)).astype("float32")
        layer = RandomEqualize(value_range=(lower, upper))
        y = layer(x)
        y = keras.ops.convert_to_numpy(y)
        self.assertTrue(np.all(y >= lower))
        self.assertTrue(np.all(y <= upper))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = RandomEqualize(value_range=(0, 255), p=1.0)
        y = layer(x)

        layer = RandomEqualize.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomEqualize(value_range=(0, 255))
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            output.numpy()

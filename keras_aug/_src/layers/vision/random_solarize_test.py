import keras
import numpy as np
from keras import backend
from keras.src import testing

from keras_aug._src.layers.vision.random_solarize import RandomSolarize


class RandomSolarizeTest(testing.TestCase):
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
        layer = RandomSolarize((0, 1), 0.5, p=1.0)
        y = layer(x)

        ref_y = TF.solarize(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), threshold=0.5
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        layer = RandomSolarize((0, 1), 0.5, p=1.0)
        y = layer(x)

        ref_y = TF.solarize(torch.tensor(x), threshold=0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        backend.set_image_data_format("channels_last")
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = RandomSolarize((0, 1), 0.5, p=0.0)
        y = layer(x)

        self.assertAllClose(y, x)

    def test_correctness_uint8(self):
        import torch
        import torchvision.transforms.v2.functional as TF

        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("uint8")
        layer = RandomSolarize((0, 255), 128, p=1.0, dtype="uint8")
        y = layer(x)

        ref_y = TF.solarize(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), threshold=128
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomSolarize((0, 255), 128)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomSolarize((0, 255), 128)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomSolarize((0, 255), 128)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomSolarize((0, 255), 128)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_data_format(self):
        # Test channels_last
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = RandomSolarize((0, 255), 128)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 32, 32, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        layer = RandomSolarize((0, 255), 128)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = RandomSolarize((0, 255), 128, p=1.0)
        y = layer(x)

        layer = RandomSolarize.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomSolarize((0, 255), 128)
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (3, 32, 32, 3))

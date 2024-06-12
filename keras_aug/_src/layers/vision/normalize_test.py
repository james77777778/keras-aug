import keras
import numpy as np
from keras import backend
from keras.src import testing

from keras_aug._src.layers.vision.normalize import Normalize


class NormalizeTest(testing.TestCase):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = Normalize((0, 1), self.mean, self.std)
        y = layer(x)

        ref_y = TF.normalize(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), self.mean, self.std
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        layer = Normalize((0, 1), self.mean, self.std)
        y = layer(x)

        ref_y = TF.normalize(torch.tensor(x), self.mean, self.std)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = Normalize((0, 255), self.mean, self.std)(x)
        self.assertEqual(y.shape, (None, None, None, 3))
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = Normalize((0, 255), self.mean, self.std)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = Normalize((0, 255), self.mean, self.std)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, 32, 32))
        y = Normalize((0, 255), self.mean, self.std)(x)
        self.assertEqual(y.shape, (None, 3, 32, 32))

    def test_model(self):
        layer = Normalize((0, 255), (0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = Normalize((0, 255), self.mean, self.std)
        y = layer(x)

        layer = Normalize.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = Normalize((0, 255), self.mean, self.std)
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (3, 32, 32, 3))

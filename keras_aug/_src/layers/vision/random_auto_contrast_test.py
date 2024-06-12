import keras
import numpy as np
from keras.src import testing

from keras_aug._src.layers.vision.random_auto_contrast import RandomAutoContrast


class RandomAutoContrastTest(testing.TestCase):
    def test_correctness(self):
        import torch
        import torchvision.transforms.functional as TF

        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = RandomAutoContrast(value_range=(0, 1), p=1.0)
        y = layer(x)

        ref_y = TF.autocontrast(torch.tensor(np.transpose(x, [0, 3, 1, 2])))
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = RandomAutoContrast(value_range=(0, 1), p=0.0)
        y = layer(x)
        self.assertAllClose(y, x)

    def test_correctness_uint8(self):
        import torch
        import torchvision.transforms.functional as TF

        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("uint8")
        layer = RandomAutoContrast(value_range=(0, 255), p=1.0, dtype="uint8")
        y = layer(x)

        ref_y = TF.autocontrast(torch.tensor(np.transpose(x, [0, 3, 1, 2])))
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomAutoContrast(value_range=(0, 1))(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomAutoContrast(value_range=(0, 1))(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomAutoContrast(value_range=(0, 255))
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = RandomAutoContrast(value_range=(0, 255), p=1.0)
        y = layer(x)

        layer = RandomAutoContrast.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomAutoContrast(value_range=(0, 255))
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (3, 32, 32, 3))

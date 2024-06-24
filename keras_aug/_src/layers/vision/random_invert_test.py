import keras
import numpy as np
from absl.testing import parameterized
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_invert import RandomInvert
from keras_aug._src.utils.test_utils import get_images


class RandomInvertTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_correctness(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        # Test channels_last
        x = get_images(dtype, "channels_last")
        layer = RandomInvert(p=1.0, dtype=dtype)
        y = layer(x)

        ref_y = TF.invert(torch.tensor(np.transpose(x, [0, 3, 1, 2])))
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        x = get_images(dtype, "channels_first")
        layer = RandomInvert(p=1.0, dtype=dtype)
        y = layer(x)

        ref_y = TF.invert(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        x = get_images(dtype, "channels_last")
        layer = RandomInvert(p=0.0, dtype=dtype)
        y = layer(x)

        self.assertAllClose(y, x)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomInvert()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomInvert()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomInvert()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = RandomInvert(p=1.0)
        y = layer(x)

        layer = RandomInvert.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomInvert()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

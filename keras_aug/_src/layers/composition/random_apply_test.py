import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing

from keras_aug._src.layers.composition.random_apply import RandomApply
from keras_aug._src.layers.vision.random_grayscale import RandomGrayscale
from keras_aug._src.layers.vision.resize import Resize
from keras_aug._src.utils.test_utils import get_images


class RandomApplyTest(testing.TestCase, parameterized.TestCase):
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

        layer = RandomApply(transforms=[RandomGrayscale(p=1.0)], p=1.0)

        x = get_images("float32", "channels_last")
        y = layer(x)

        ref_y = TF.rgb_to_grayscale(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), num_output_channels=3
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test p=0.0
        layer = RandomApply(transforms=[RandomGrayscale(p=1.0)], p=0.0)
        y = layer(x)

        self.assertAllClose(y, x)

    def test_shape(self):
        layer = RandomApply(transforms=RandomGrayscale(p=1.0))

        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = layer(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = layer(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

        # Test deterministic shape
        transform = Resize((16, 16))
        layer = RandomApply(transforms=transform)
        x = keras.KerasTensor((None, 16, 16, 3))
        y = layer(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

    def test_model(self):
        layer = RandomApply(transforms=RandomGrayscale(p=1.0))
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = RandomApply(transforms=RandomGrayscale(p=1.0), p=1.0)
        y = layer(x)

        layer = RandomApply.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomApply(transforms=RandomGrayscale(p=1.0))
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

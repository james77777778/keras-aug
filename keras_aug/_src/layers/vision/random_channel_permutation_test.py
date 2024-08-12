import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras import ops
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_channel_permutation import (
    RandomChannelPermutation,
)
from keras_aug._src.testing.test_case import TestCase
from keras_aug._src.utils.test_utils import get_images


class FixedRandomChannelPermutation(RandomChannelPermutation):
    def get_params(self, batch_size, images=None, **kwargs):
        zeros = ops.zeros([batch_size], "int32")
        ones = ops.ones([batch_size], "int32")
        twos = ops.ones([batch_size], "int32") * 2
        return ops.stack([ones, twos, zeros], axis=-1)


class RandomChannelPermutationTest(TestCase):
    @parameterized.named_parameters(
        named_product(dtype=["float32", "mixed_bfloat16", "uint8"])
    )
    def test_correctness(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF
        from keras.src.backend.torch import convert_to_tensor

        np.random.seed(42)

        # Test channels_last
        x = get_images(dtype, "channels_last")
        layer = FixedRandomChannelPermutation(3, dtype=dtype)
        y = layer(x)

        ref_y = TF.permute_channels(
            convert_to_tensor(np.transpose(x, [0, 3, 1, 2])), [1, 2, 0]
        )
        ref_y = torch.permute(ref_y, (0, 2, 3, 1))
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images(dtype, "channels_first")
        layer = FixedRandomChannelPermutation(3, dtype=dtype)
        y = layer(x)

        ref_y = TF.permute_channels(convert_to_tensor(x), [1, 2, 0])
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomChannelPermutation(3)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomChannelPermutation(3)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomChannelPermutation(3)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomChannelPermutation(3)
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_data_format(self):
        # Test channels_last
        x = get_images("float32", "channels_last")
        layer = RandomChannelPermutation(3)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 32, 32, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images("float32", "channels_first")
        layer = RandomChannelPermutation(3)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 32))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedRandomChannelPermutation(3)
        y = layer(x)

        layer = FixedRandomChannelPermutation.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomChannelPermutation(3)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

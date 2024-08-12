import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.to_dtype import ToDType
from keras_aug._src.testing.test_case import TestCase
from keras_aug._src.utils.test_utils import get_images


class ToDTypeTest(TestCase):
    @parameterized.named_parameters(
        named_product(
            from_dtype=["uint8", "int16", "int32", "bfloat16", "float32"],
            to_dtype=["uint8", "int16", "bfloat16", "float32"],
            scale=[True, False],
        )
    )
    def test_correctness(self, from_dtype, to_dtype, scale):
        import torch
        import torchvision.transforms.v2.functional as TF
        from keras.src.backend.torch import convert_to_tensor
        from keras.src.backend.torch import to_torch_dtype

        # Test channels_last
        x = get_images(from_dtype, "channels_last")
        layer = ToDType(to_dtype, scale)
        y = layer(x)

        ref_y = TF.to_dtype(
            convert_to_tensor(np.transpose(x, [0, 3, 1, 2])),
            dtype=to_torch_dtype(to_dtype),
            scale=scale,
        )
        ref_y = torch.permute(ref_y, (0, 2, 3, 1))
        self.assertDType(y, to_dtype)
        if from_dtype == "bfloat16" and to_dtype in ("uint8", "int16"):
            return
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = ToDType("float32", scale=True)(x)
        self.assertEqual(y.shape, (None, None, None, 3))
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = ToDType("float32", scale=True)(x)
        self.assertEqual(y.shape, (None, 3, None, None))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = ToDType("float32", scale=True)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, 32, 32))
        y = ToDType("float32", scale=True)(x)
        self.assertEqual(y.shape, (None, 3, 32, 32))

    def test_model(self):
        layer = ToDType("float32", scale=True)
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = ToDType("float32", scale=True)
        y = layer(x)

        layer = ToDType.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = ToDType("float32", scale=True)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

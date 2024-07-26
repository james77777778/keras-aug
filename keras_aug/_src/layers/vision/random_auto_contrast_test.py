import keras
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_auto_contrast import RandomAutoContrast
from keras_aug._src.utils.test_utils import get_images


class RandomAutoContrastTest(testing.TestCase, parameterized.TestCase):
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

        if dtype == "uint8":
            atol = 1
        else:
            atol = 1e-6
        x = get_images(dtype, "channels_first")
        layer = RandomAutoContrast(
            p=1.0, dtype=dtype, data_format="channels_first"
        )
        y = layer(x)

        ref_y = TF.autocontrast(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y, atol=atol)

        # Test p=0.0
        x = get_images(dtype, "channels_last")
        layer = RandomAutoContrast(p=0.0, dtype=dtype)
        y = layer(x)
        self.assertDType(y, dtype)
        self.assertAllClose(y, x)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomAutoContrast()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomAutoContrast()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomAutoContrast()
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = RandomAutoContrast(p=1.0)
        y = layer(x)

        layer = RandomAutoContrast.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomAutoContrast()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

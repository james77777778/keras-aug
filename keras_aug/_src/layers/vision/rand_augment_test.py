import keras
from absl.testing import parameterized
from keras import backend
from keras.src import testing

from keras_aug._src.layers.vision.rand_augment import RandAugment
from keras_aug._src.utils.test_utils import get_images


class RandAugmentTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    # TODO: Add correctness test

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandAugment()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandAugment()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        # Test dynamic shape
        layer = RandAugment()
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

        # Test static shape
        layer = RandAugment()
        inputs = keras.layers.Input(shape=[32, 32, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 32, 32, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = RandAugment()
        y = layer(x)

        layer = RandAugment.from_config(layer.get_config())
        y2 = layer(x)
        self.assertEqual(y.shape, y2.shape)

        # Test `p=0.0`
        layer = RandAugment(p=0.0)
        y = layer(x)

        layer = RandAugment.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, x)
        self.assertAllClose(y2, x)
        self.assertEqual(y.shape, y2.shape)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandAugment()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

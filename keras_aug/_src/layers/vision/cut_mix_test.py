import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.cut_mix import CutMix
from keras_aug._src.utils.test_utils import get_images


class FixedCutMix(CutMix):
    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        top = ops.numpy.zeros([batch_size])
        left = ops.numpy.zeros([batch_size])
        bottom = ops.numpy.ones([batch_size]) * 10
        right = ops.numpy.ones([batch_size]) * 10
        lam = ops.numpy.ones([batch_size]) * 0.5
        return dict(top=top, bottom=bottom, left=left, right=right, lam=lam)


class CutMixTest(testing.TestCase, parameterized.TestCase):
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
        # Test channels_last
        images = get_images(dtype, "channels_last")
        labels = np.array([0, 1], "float32")
        inputs = {"images": images, "labels": labels}
        layer = FixedCutMix(num_classes=2, dtype=dtype)
        outputs = layer(inputs)

        self.assertDType(outputs["images"], dtype)
        self.assertAllClose(
            outputs["images"][0, 0:10, 0:10, :], images[1, 0:10, 0:10, :]
        )
        self.assertAllClose(
            outputs["images"][0, 10:, 10:, :], images[0, 10:, 10:, :]
        )
        self.assertAllClose(outputs["labels"], [[0.5, 0.5], [0.5, 0.5]])

        # Test channels_first
        backend.set_image_data_format("channels_first")
        images = get_images(dtype, "channels_first")
        labels = np.array([0, 1], "float32")
        inputs = {"images": images, "labels": labels}
        layer = FixedCutMix(num_classes=2, dtype=dtype)
        outputs = layer(inputs)

        self.assertDType(outputs["images"], dtype)
        self.assertAllClose(
            outputs["images"][0, :, 0:10, 0:10], images[1, :, 0:10, 0:10]
        )
        self.assertAllClose(
            outputs["images"][0, :, 10:, 10:], images[0, :, 10:, 10:]
        )
        self.assertAllClose(outputs["labels"], [[0.5, 0.5], [0.5, 0.5]])

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = CutMix()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = CutMix()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = CutMix()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedCutMix()
        y = layer(x)

        layer = FixedCutMix.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = CutMix()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

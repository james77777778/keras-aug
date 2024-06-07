import keras
import numpy as np
from keras.src import testing

from keras_aug._src.layers.vision.rescale import Rescale


class RescaleTest(testing.TestCase):
    def test_correctness(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("uint8")
        layer = Rescale(scale=2, offset=0.5)
        y = layer(x)

        ref_y = x * 2.0 + 0.5
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = Rescale(scale=2, offset=0.5)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = Rescale(scale=2, offset=0.5)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = Rescale(scale=2, offset=0.5)
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = Rescale(scale=2, offset=0.5)
        y = layer(x)

        layer = Rescale.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = Rescale(scale=2, offset=0.5)
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            output.numpy()

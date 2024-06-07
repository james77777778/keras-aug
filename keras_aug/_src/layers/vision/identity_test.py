import keras
import numpy as np
from keras.src import testing

from keras_aug._src.layers.vision.identity import Identity


class IdentityTest(testing.TestCase):
    def test_correctness(self):
        x = np.random.uniform(0, 1, (2, 32, 32, 3))
        layer = Identity()
        y = layer(x)
        self.assertAllClose(y, x)

        x = {
            "images": np.random.uniform(0, 1, (2, 32, 32, 3)),
            "bounding_boxes": {
                "boxes": np.random.uniform(0, 1, (2, 10, 4)),
                "classes": np.random.uniform(0, 1, (2, 10, 5)),
            },
            "segmentation_masks": np.random.uniform(0, 1, (2, 32, 32, 1)),
            "keypoints": np.random.uniform(0, 1, (2, 10, 17)),
        }
        y = layer(x)
        self.assertAllClose(y["images"], x["images"])
        self.assertAllClose(
            y["bounding_boxes"]["boxes"], x["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            y["bounding_boxes"]["classes"], x["bounding_boxes"]["classes"]
        )
        self.assertAllClose(y["segmentation_masks"], x["segmentation_masks"])
        self.assertAllClose(y["keypoints"], x["keypoints"])

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = Identity()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = Identity()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = Identity()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = Identity()
        y = layer(x)

        layer = Identity.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = Identity()
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            output.numpy()

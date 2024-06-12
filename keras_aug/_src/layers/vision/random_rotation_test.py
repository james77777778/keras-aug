import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_rotation import RandomRotation


class FixedRandomRotation(RandomRotation):
    def get_params(self, batch_size, images=None, **kwargs):
        return dict(angle=keras.ops.array([10.0]))


class RandomRotationTest(testing.TestCase, parameterized.TestCase):
    regular_args = dict(degree=[-10, 10])

    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    @parameterized.named_parameters(
        named_product(
            interpolation=["nearest", "bilinear"],
        )
    )
    def test_correctness(self, interpolation):
        # Test channels_last
        np.random.seed(42)
        x = np.random.uniform(0, 1, (1, 32, 32, 3)).astype("float32")
        layer = FixedRandomRotation(interpolation=interpolation)
        y = layer(x)

        # TODO: It is difficult to be consistent with `TF.affine`
        self.assertEqual(y.shape, (1, 32, 32, 3))
        self.assertNotAllClose(y, x)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        np.random.seed(42)
        x = np.random.uniform(0, 1, (1, 3, 32, 32)).astype("float32")
        layer = FixedRandomRotation(interpolation=interpolation)
        y = layer(x)

        # TODO: It is difficult to be consistent with `TF.affine`
        self.assertEqual(y.shape, (1, 3, 32, 32))
        self.assertNotAllClose(y, x)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomRotation(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomRotation(**self.regular_args)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        # Test dynamic shape
        layer = RandomRotation(**self.regular_args)
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

        # Test static shape
        layer = RandomRotation(**self.regular_args)
        inputs = keras.layers.Input(shape=[32, 32, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 32, 32, 3))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = RandomRotation(degree=[10, 10])
        y = layer(x)

        layer = RandomRotation.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomRotation(**self.regular_args)
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (3, 32, 32, 3))

    def test_augment_bounding_box(self):
        # Test full bounding boxes
        images = np.zeros([1, 32, 32, 3]).astype("float32")
        boxes = {
            "boxes": np.array(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                ],
                "float32",
            ),
            "classes": np.array([[0, 0, 0, 0]], "float32"),
        }
        input = {"images": images, "bounding_boxes": boxes}
        layer = FixedRandomRotation(bounding_box_format="rel_xyxy")

        output = layer(input)
        self.assertAllClose(output["bounding_boxes"]["boxes"], boxes["boxes"])
        self.assertAllClose(
            output["bounding_boxes"]["classes"], boxes["classes"]
        )

        # Test arbitrary bounding boxes
        images = np.zeros([1, 32, 32, 3]).astype("float32")
        boxes = {
            "boxes": np.array(
                [
                    [
                        [3, 4, 17, 18],
                        [10, 12, 16, 19],
                        [0, 0, 1, 1],
                        [15, 12, 17, 20],
                    ],
                ],
                "float32",
            ),
            "classes": np.array([[0, 1, 2, 3]], "float32"),
        }
        input = {"images": images, "bounding_boxes": boxes}
        layer = FixedRandomRotation(bounding_box_format="xyxy")

        output = layer(input)
        expected_boxes = {
            "boxes": np.array(
                [
                    [
                        [2.850203, 1.924881, 19.068584, 18.143265],
                        [9.57021, 11.01888, 16.694592, 18.954426],
                        [2.847799, 0.0, 4.006255, 0.0],
                        [14.3206, 11.887121, 17.679401, 20.11288],
                    ],
                ],
                "float32",
            ),
            "classes": np.array([[0, 1, -1, 3]], "float32"),
        }
        self.assertAllClose(
            output["bounding_boxes"]["boxes"], expected_boxes["boxes"]
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"], expected_boxes["classes"]
        )

    def test_augment_segmentation_mask(self):
        num_classes = 8
        images_shape = (1, 32, 32, 3)
        masks_shape = (1, 32, 32, 1)
        images = np.random.uniform(size=images_shape).astype("float32")
        masks = np.random.randint(2, size=masks_shape) * (num_classes - 1)
        inputs = {"images": images, "segmentation_masks": masks}

        layer = FixedRandomRotation()
        output = layer(inputs)
        self.assertEqual(output["segmentation_masks"].shape, masks_shape)
        # TODO: Verify the correctness

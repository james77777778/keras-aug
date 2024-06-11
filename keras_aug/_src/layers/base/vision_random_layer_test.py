import keras
import numpy as np
import pytest
from keras import backend
from keras.src import testing

from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


class RandomAddLayer(VisionRandomLayer):
    def __init__(self, add_range=(0.0, 1.0), fixed_value=None, **kwargs):
        has_generator = fixed_value is None
        super().__init__(has_generator=has_generator, **kwargs)
        self.add_range = add_range
        self.fixed_value = fixed_value

    def get_params(self, batch_size, **kwargs):
        ops = self.backend
        if self.fixed_value:
            return ops.numpy.ones((batch_size,)) * self.fixed_value
        return ops.random.uniform(
            (batch_size,),
            minval=self.add_range[0],
            maxval=self.add_range[1],
            seed=self.random_generator,
        )

    def augment_images(self, images, transformations, **kwargs):
        return images + transformations[:, None, None, None]

    def augment_labels(self, labels, transformations, **kwargs):
        return labels + transformations[:, None]

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return {
            "boxes": bounding_boxes["boxes"] + transformations[:, None, None],
            "classes": bounding_boxes["classes"] + transformations[:, None],
        }

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints + transformations[:, None, None]

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks + transformations[:, None, None, None]


class AssertionLayer(VisionRandomLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_params(
        self,
        batch_size,
        images=None,
        labels=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_masks=None,
        **kwargs,
    ):
        ops = self.backend
        assert ops.is_tensor(images)
        assert ops.is_tensor(labels)
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(keypoints)
        assert ops.is_tensor(segmentation_masks)
        return ops.random.uniform([batch_size], seed=self.random_generator)

    def augment_images(
        self,
        images,
        transformations=None,
        bounding_boxes=None,
        labels=None,
        **kwargs,
    ):
        ops = self.backend
        assert ops.is_tensor(images)
        assert ops.is_tensor(transformations)
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(labels)
        return images

    def augment_labels(
        self,
        labels,
        transformations=None,
        bounding_boxes=None,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        ops = self.backend
        assert ops.is_tensor(labels)
        assert ops.is_tensor(transformations)
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(images)
        assert ops.is_tensor(raw_images)
        return labels

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformations=None,
        labels=None,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        ops = self.backend
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(transformations)
        assert ops.is_tensor(labels)
        assert ops.is_tensor(images)
        assert ops.is_tensor(raw_images)
        return bounding_boxes

    def augment_keypoints(
        self,
        keypoints,
        transformations=None,
        labels=None,
        bounding_boxes=None,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        ops = self.backend
        assert ops.is_tensor(keypoints)
        assert ops.is_tensor(transformations)
        assert ops.is_tensor(labels)
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(images)
        assert ops.is_tensor(raw_images)
        return keypoints

    def augment_segmentation_masks(
        self,
        segmentation_masks,
        transformations=None,
        labels=None,
        bounding_boxes=None,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        ops = self.backend
        assert ops.is_tensor(segmentation_masks)
        assert ops.is_tensor(transformations)
        assert ops.is_tensor(labels)
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(images)
        assert ops.is_tensor(raw_images)
        return segmentation_masks

    def augment_custom_annotations(
        self,
        custom_annotations,
        transformations=None,
        labels=None,
        bounding_boxes=None,
        images=None,
        raw_images=None,
    ):
        ops = self.backend
        assert ops.is_tensor(custom_annotations)
        assert ops.is_tensor(transformations)
        assert ops.is_tensor(labels)
        assert ops.is_tensor(bounding_boxes["boxes"])
        assert ops.is_tensor(bounding_boxes["classes"])
        assert ops.is_tensor(images)
        assert ops.is_tensor(raw_images)
        return custom_annotations


class VisionRandomLayerTest(testing.TestCase):
    def test_single_image(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        output = add_layer(image)

        self.assertAllClose(image + 2.0, output)

    def test_dict(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        output = add_layer({"images": image})

        self.assertIsInstance(output, dict)

    def test_casting(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        images = np.ones((2, 8, 8, 3), dtype="uint8")
        output = add_layer(images)

        self.assertAllClose(
            np.ones((2, 8, 8, 3), dtype="float32") * 3.0, output
        )

    def test_batch_images(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        output = add_layer(images)

        diff = output - images
        # Make sure the first image and second image get different augmentation
        self.assertNotAllClose(diff[0], diff[1])

    def test_image_and_label(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        image = np.random.random(size=(8, 8, 3)).astype("float32")
        label = np.random.random(size=(1,)).astype("float32")

        output = add_layer({"images": image, "labels": label})
        expected_output = {"images": image + 2.0, "labels": label + 2.0}
        self.assertAllClose(output["images"], expected_output["images"])
        self.assertAllClose(output["labels"], expected_output["labels"])

    def test_batch_images_and_labels(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        labels = np.random.random(size=(2, 1)).astype("float32")
        output = add_layer({"images": images, "labels": labels})

        image_diff = output["images"] - images
        label_diff = output["labels"] - labels
        # Make sure the first image and second image get different augmentation
        self.assertNotAllClose(image_diff[0], image_diff[1])
        self.assertNotAllClose(label_diff[0], label_diff[1])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason=("Requires tensorflow backend"),
    )
    def test_unmodified_data(self):
        import tensorflow as tf

        add_layer = RandomAddLayer(fixed_value=0.5)
        images = np.random.random(size=(8, 8, 3)).astype("float32")
        filenames = tf.constant("/path/to/first.jpg")
        inputs = {"images": images, "filenames": filenames}
        output = add_layer(inputs)
        self.assertEqual(output["filenames"], filenames)

    def test_image_and_bbox(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        images = np.random.random(size=(8, 8, 8, 3)).astype("float32")
        bounding_boxes = {
            "boxes": np.random.random(size=(8, 3, 4)).astype("float32"),
            "classes": np.random.random(size=(8, 3)).astype("float32"),
        }
        keypoints = np.random.random(size=(8, 5, 2)).astype("float32")
        segmentation_mask = np.random.random(size=(8, 8, 8, 1)).astype(
            "float32"
        )

        output = add_layer(
            {
                "images": images,
                "bounding_boxes": bounding_boxes,
                "keypoints": keypoints,
                "segmentation_masks": segmentation_mask,
            }
        )
        expected_output = {
            "images": images + 2.0,
            "bounding_boxes": {
                "boxes": bounding_boxes["boxes"] + 2.0,
                "classes": bounding_boxes["classes"] + 2.0,
            },
            "keypoints": keypoints + 2.0,
            "segmentation_masks": segmentation_mask + 2.0,
        }

        self.assertAllClose(output["images"], expected_output["images"])
        self.assertAllClose(output["keypoints"], expected_output["keypoints"])
        self.assertAllClose(
            output["bounding_boxes"]["boxes"],
            expected_output["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"],
            expected_output["bounding_boxes"]["classes"],
        )
        self.assertAllClose(
            output["segmentation_masks"], expected_output["segmentation_masks"]
        )

    def test_batch_images_and_bboxes(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        bounding_boxes = {
            "boxes": np.random.random(size=(2, 3, 4)).astype("float32"),
            "classes": np.random.random(size=(2, 3)).astype("float32"),
        }
        keypoints = np.random.random(size=(2, 5, 2)).astype("float32")
        segmentation_masks = np.random.random(size=(2, 8, 8, 1)).astype(
            "float32"
        )

        output = add_layer(
            {
                "images": images,
                "bounding_boxes": bounding_boxes,
                "keypoints": keypoints,
                "segmentation_masks": segmentation_masks,
            }
        )
        bounding_boxes_diff = (
            output["bounding_boxes"]["boxes"] - bounding_boxes["boxes"]
        )
        keypoints_diff = output["keypoints"] - keypoints
        segmentation_mask_diff = (
            output["segmentation_masks"] - segmentation_masks
        )
        self.assertNotAllClose(bounding_boxes_diff[0], bounding_boxes_diff[1])
        self.assertNotAllClose(keypoints_diff[0], keypoints_diff[1])
        self.assertNotAllClose(
            segmentation_mask_diff[0], segmentation_mask_diff[1]
        )

        # Build Functional model
        inputs = {
            "images": keras.Input([8, 8, 3]),
            "bounding_boxes": {
                "boxes": keras.Input([3, 4]),
                "classes": keras.Input([3]),
            },
            "keypoints": keras.Input([5, 2]),
            "segmentation_masks": keras.Input([8, 8, 1]),
        }
        outputs = add_layer(inputs)
        model = keras.Model(inputs, outputs)

        output = model.predict(
            {
                "images": images,
                "bounding_boxes": bounding_boxes,
                "keypoints": keypoints,
                "segmentation_masks": segmentation_masks,
            }
        )
        bounding_boxes_diff = (
            output["bounding_boxes"]["boxes"] - bounding_boxes["boxes"]
        )
        keypoints_diff = output["keypoints"] - keypoints
        segmentation_mask_diff = (
            output["segmentation_masks"] - segmentation_masks
        )
        self.assertNotAllClose(bounding_boxes_diff[0], bounding_boxes_diff[1])
        self.assertNotAllClose(keypoints_diff[0], keypoints_diff[1])
        self.assertNotAllClose(
            segmentation_mask_diff[0], segmentation_mask_diff[1]
        )

    def test_all_data_in_model(self):
        add_layer = RandomAddLayer()
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        bounding_boxes = {
            "boxes": np.random.random(size=(2, 3, 4)).astype("float32"),
            "classes": np.random.random(size=(2, 3)).astype("float32"),
        }
        keypoints = np.random.random(size=(2, 5, 2)).astype("float32")
        segmentation_masks = np.random.random(size=(2, 8, 8, 1)).astype(
            "float32"
        )

        # Build Functional model
        inputs = {
            "images": keras.Input([8, 8, 3]),
            "bounding_boxes": {
                "boxes": keras.Input([3, 4]),
                "classes": keras.Input([3]),
            },
            "keypoints": keras.Input([5, 2]),
            "segmentation_masks": keras.Input([8, 8, 1]),
        }
        outputs = add_layer(inputs)
        model = keras.Model(inputs, outputs)

        output = model.predict(
            {
                "images": images,
                "bounding_boxes": bounding_boxes,
                "keypoints": keypoints,
                "segmentation_masks": segmentation_masks,
            }
        )
        bounding_boxes_diff = (
            output["bounding_boxes"]["boxes"] - bounding_boxes["boxes"]
        )
        keypoints_diff = output["keypoints"] - keypoints
        segmentation_mask_diff = (
            output["segmentation_masks"] - segmentation_masks
        )
        self.assertNotAllClose(bounding_boxes_diff[0], bounding_boxes_diff[1])
        self.assertNotAllClose(keypoints_diff[0], keypoints_diff[1])
        self.assertNotAllClose(
            segmentation_mask_diff[0], segmentation_mask_diff[1]
        )

    def test_unbatched_all_data(self):
        add_layer = RandomAddLayer(fixed_value=2.0)
        images = np.random.random(size=(8, 8, 3)).astype("float32")
        bounding_boxes = {
            "boxes": np.random.random(size=(3, 4)).astype("float32"),
            "classes": np.random.random(size=(3)).astype("float32"),
        }
        keypoints = np.random.random(size=(5, 2)).astype("float32")
        segmentation_masks = np.random.random(size=(8, 8, 1)).astype("float32")
        input = {
            "images": images,
            "bounding_boxes": bounding_boxes,
            "keypoints": keypoints,
            "segmentation_masks": segmentation_masks,
        }

        output = add_layer(input, training=True)
        expected_output = {
            "images": images + 2.0,
            "bounding_boxes": {
                "boxes": bounding_boxes["boxes"] + 2.0,
                "classes": bounding_boxes["classes"] + 2.0,
            },
            "keypoints": keypoints + 2.0,
            "segmentation_masks": segmentation_masks + 2.0,
        }

        self.assertAllClose(output["images"], expected_output["images"])
        self.assertAllClose(output["keypoints"], expected_output["keypoints"])
        self.assertAllClose(
            output["bounding_boxes"]["boxes"],
            expected_output["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"],
            expected_output["bounding_boxes"]["classes"],
        )
        self.assertAllClose(
            output["segmentation_masks"], expected_output["segmentation_masks"]
        )

    def test_all_data_assertion(self):
        images = np.random.random(size=(2, 8, 8, 3)).astype("float32")
        labels = np.squeeze(np.eye(10)[np.array([0, 1]).reshape(-1)])
        bounding_boxes = {
            "boxes": np.random.random(size=(2, 3, 4)).astype("float32"),
            "classes": np.random.random(size=(2, 3)).astype("float32"),
        }
        keypoints = np.random.random(size=(2, 5, 2)).astype("float32")
        segmentation_masks = np.random.random(size=(2, 8, 8, 1)).astype(
            "float32"
        )
        custom_annotations = np.random.random(size=(2, 1)).astype("float32")
        assertion_layer = AssertionLayer()

        _ = assertion_layer(
            {
                "images": images,
                "labels": labels,
                "bounding_boxes": bounding_boxes,
                "keypoints": keypoints,
                "segmentation_masks": segmentation_masks,
                "custom_annotations": custom_annotations,
            }
        )

        # Assertions are at AssertionLayer's methods

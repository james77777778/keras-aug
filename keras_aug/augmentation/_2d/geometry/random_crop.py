import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomCrop(VectorizedBaseRandomLayer):
    """Randomly crops the input images.

    This layer will randomly choose a location to crop images down to
    ``(height, width)``. If an input image is smaller than the
    ``(height, width)``, the input will be resized and cropped to return the
    largest possible window in the image that matches the target aspect ratio.

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        interpolation (str, optional): The interpolation mode. Supported values:
            ``"nearest", "bilinear"``. Defaults to `"bilinear"`.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.resize_bilinear = keras.layers.Resizing(self.height, self.width)
        self.resize_nearest = keras.layers.Resizing(
            self.height, self.width, interpolation="nearest"
        )
        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True
        self.force_output_dense_segmentation_masks = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        tops = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        lefts = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        return {"crop_tops": tops, "crop_lefts": lefts}

    def compute_ragged_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=(self.height, self.width, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        heights, widths = augmentation_utils.get_images_shape(images)
        h_diffs = heights - self.height
        w_diffs = widths - self.width
        # broadcast
        h_diffs = h_diffs[:, tf.newaxis, tf.newaxis, :]
        w_diffs = w_diffs[:, tf.newaxis, tf.newaxis, :]
        images = tf.where(
            tf.math.logical_and(h_diffs >= 0, w_diffs >= 0),
            self.crop_images(images, transformations),
            self.resize_images(images, interpolation=self.interpolation),
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformations,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCrop(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )
        heights, widths = augmentation_utils.get_images_shape(raw_images)
        h_diffs = heights - self.height
        w_diffs = widths - self.width
        # broadcast
        h_diffs = h_diffs[:, tf.newaxis, :]
        w_diffs = w_diffs[:, tf.newaxis, :]
        boxes = tf.where(
            tf.math.logical_and(h_diffs >= 0, w_diffs >= 0),
            self.crop_bounding_boxes(
                raw_images, bounding_boxes["boxes"], transformations
            ),
            self.resize_bounding_boxes(
                raw_images,
                bounding_boxes["boxes"],
            ),
        )
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            image_shape=(self.height, self.width, None),
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            image_shape=(self.height, self.width, None),
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(
            segmentation_masks
        )
        h_diffs = heights - self.height
        w_diffs = widths - self.width
        # broadcast
        h_diffs = h_diffs[:, tf.newaxis, tf.newaxis, :]
        w_diffs = w_diffs[:, tf.newaxis, tf.newaxis, :]
        segmentation_masks = tf.where(
            tf.math.logical_and(h_diffs >= 0, w_diffs >= 0),
            self.crop_images(segmentation_masks, transformations),
            self.resize_images(segmentation_masks, interpolation="nearest"),
        )
        return segmentation_masks

    def crop_images(self, images, transformations):
        batch_size = tf.shape(images)[0]
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        tops = transformations["crop_tops"]
        lefts = transformations["crop_lefts"]
        x1s = lefts * (widths - self.width)
        y1s = tops * (heights - self.height)
        x2s = x1s + self.width
        y2s = y1s + self.height
        # normalize
        x1s /= widths
        y1s /= heights
        x2s /= widths
        y2s /= heights
        boxes = tf.concat([y1s, x1s, y2s, x2s], axis=-1)
        images = tf.image.crop_and_resize(
            images,
            boxes,
            tf.range(batch_size),
            [self.height, self.width],
            method="nearest",
        )
        return tf.cast(images, dtype=self.compute_dtype)

    def resize_images(self, images, interpolation="bilinear"):
        if interpolation == "bilinear":
            images = self.resize_bilinear(images)
        elif interpolation == "nearest":
            images = self.resize_nearest(images)
        else:
            raise ValueError(f"Unsupported interpolation: {interpolation}")
        return tf.cast(images, dtype=self.compute_dtype)

    def crop_bounding_boxes(self, images, boxes, transformation):
        tops = transformation["crop_tops"]
        lefts = transformation["crop_lefts"]
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )

        # compute offsets for xyxy bounding_boxes
        top_offsets = tf.cast(
            tf.math.round(tops * (heights - self.height)),
            dtype=self.compute_dtype,
        )
        left_offsets = tf.cast(
            tf.math.round(lefts * (widths - self.width)),
            dtype=self.compute_dtype,
        )
        x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=-1)
        x1s -= tf.expand_dims(left_offsets, axis=1)
        y1s -= tf.expand_dims(top_offsets, axis=1)
        x2s -= tf.expand_dims(left_offsets, axis=1)
        y2s -= tf.expand_dims(top_offsets, axis=1)
        outputs = tf.concat([x1s, y1s, x2s, y2s], axis=-1)
        return outputs

    def resize_bounding_boxes(self, images, boxes):
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        x_scale = tf.cast(self.width / widths, dtype=self.compute_dtype)
        y_scale = tf.cast(self.height / heights, dtype=self.compute_dtype)
        x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=-1)
        outputs = tf.concat(
            [
                x1s * x_scale[:, tf.newaxis, :],
                y1s * y_scale[:, tf.newaxis, :],
                x2s * x_scale[:, tf.newaxis, :],
                y2s * y_scale[:, tf.newaxis, :],
            ],
            axis=-1,
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
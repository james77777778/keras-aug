import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class Resize(VectorizedBaseRandomLayer):
    """Resizes the images.

    Resize will resize the images to ``(height, width)``. The aspect ratio might
    be different between the result images and the initial images.

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        interpolation (str, optional): The interpolation mode.
            Supported values: ``"nearest", "bilinear"``.
            Defaults to `"bilinear"`.
        antialias (bool, optional): Whether to use antialias. Defaults to
            ``False``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.
    """  # noqa: E501

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        antialias=False,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                "`height` and `width` must be integer. Received: "
                f"`height`={height} `width`={width} "
            )
        self.height = height
        self.width = width
        self.interpolation = preprocessing_utils.get_interpolation(
            interpolation
        )
        self.antialias = antialias
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True
        self.force_output_dense_segmentation_masks = True

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        # resize
        images = tf.image.resize(
            images,
            size=(self.height, self.width),
            method=self.interpolation,
            antialias=self.antialias,
        )
        return tf.cast(images, dtype=self.compute_dtype)

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
                "`Resize()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`Resize(..., bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=raw_images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=images,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        if isinstance(segmentation_masks, tf.RaggedTensor):
            inputs = {
                augmentation_utils.SEGMENTATION_MASKS: segmentation_masks,
            }
            segmentation_masks = tf.vectorized_map(
                self.augment_segmentation_mask_single,
                inputs,
            )
        else:
            # resize
            segmentation_masks = tf.image.resize(
                segmentation_masks,
                size=(self.height, self.width),
                method="nearest",
            )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

    def augment_segmentation_mask_single(self, inputs):
        segmentation_mask = inputs.get(
            augmentation_utils.SEGMENTATION_MASKS, None
        )
        # resize
        segmentation_mask = tf.image.resize(
            segmentation_mask,
            size=(self.height, self.width),
            method="nearest",
        )
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

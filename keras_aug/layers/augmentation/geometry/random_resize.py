import typing

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomResize(VectorizedBaseRandomLayer):
    """Randomly resizes the images in a batch manner.

    This layer is useful for multi-scale training.

    Notes:
        The aspect ratio might be different from the original images.

    Args:
        heights (list(int)): The heights to be sampled for the result image.
        widths (list(int), optional): The widths to be sampled for the result
            image. Defaults to ``None``. If setting ``None``, ``widths`` will
            be the same as ``heights``.
        interpolation (str, optional): The interpolation mode.
            Supported values: ``"nearest", "bilinear"``. Defaults to
            ``"bilinear"``.
        antialias (bool, optional): Whether to use antialias. Defaults to
            ``False``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to ``None``.
    """  # noqa: E501

    def __init__(
        self,
        heights,
        widths=None,
        interpolation="bilinear",
        antialias=False,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.heights = heights
        if widths is None:
            self.widths = heights
        else:
            self.widths = widths
        if not isinstance(self.heights, typing.Sequence) or not isinstance(
            self.widths, typing.Sequence
        ):
            raise ValueError(
                "RandomResize expects `heights` and `widths` to be a list of "
                f"int. Received: `heights`={heights}, `widths`={widths}"
            )
        if len(self.heights) != len(self.widths):
            raise ValueError(
                "RandomResize expects `heights` and `widths` to be same "
                f"length. Received: `heights`={heights}, `widths`={widths}"
            )
        self.interpolation = preprocessing_utils.get_interpolation(
            interpolation
        )
        self.antialias = antialias
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self._heights = tf.convert_to_tensor(self.heights, dtype=tf.int32)
        self._widths = tf.convert_to_tensor(self.widths, dtype=tf.int32)

        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # sample 1 height and width for the batch
        indice = self._random_generator.random_uniform(
            shape=(1,),
            minval=0,
            maxval=len(self.heights),
            dtype=tf.int32,
        )
        height = tf.gather(self._heights, indice)
        width = tf.gather(self._widths, indice)
        scaled_sizes = tf.stack([height, width], axis=-1)
        scaled_sizes = tf.tile(scaled_sizes, multiples=(batch_size, 1))
        return {"scaled_sizes": scaled_sizes}

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
        scaled_sizes = transformations["scaled_sizes"]

        # currently, only support same size within dense batch
        # this layer might produce varying size with ragged batch
        new_height = scaled_sizes[0][0]
        new_width = scaled_sizes[0][1]
        images = tf.image.resize(
            images,
            size=(new_height, new_width),
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

        # resize
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

    def augment_ragged_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        segmentation_mask = tf.expand_dims(segmentation_mask, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        segmentation_mask = self.augment_segmentation_masks(
            segmentation_masks=segmentation_mask,
            transformations=transformation,
            **kwargs,
        )
        return tf.squeeze(segmentation_mask, axis=0)

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        scaled_size = transformations["scaled_sizes"]
        new_height = scaled_size[0][0]
        new_width = scaled_size[0][1]
        # resize
        segmentation_masks = tf.image.resize(
            segmentation_masks,
            size=(new_height, new_width),
            method="nearest",
        )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "heights": self.heights,
                "widths": self.widths,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

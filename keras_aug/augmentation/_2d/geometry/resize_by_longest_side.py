import collections.abc

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class ResizeByLongestSide(VectorizedBaseRandomLayer):
    """Resize images so that longest side is equal to ``max_size``, keeping the
    aspect ratio of the initial images.

    This layer produces outputs of the same ``max_size`` within a batch but may
    varying ``max_size`` across different batches if ``max_size`` is a list.

    Args:
        max_size (int|list(int)): The size of the longest side of result image.
            When using a list, the ``max_size`` will be randomly selected from
            the list.
        interpolation (str, optional): The interpolation mode. Supported values:
            ``"nearest", "bilinear"``. Defaults to `"bilinear"`.
        antialias (bool, optional): Whether to use antialias. Defaults to
            ``False``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `Albumentations <https://github.com/albumentations-team/albumentations>`_
    """  # noqa: E501

    def __init__(
        self,
        max_size,
        interpolation="bilinear",
        antialias=False,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(max_size, collections.abc.Sequence):
            if not isinstance(max_size[0], int):
                raise ValueError(
                    "`max_size` must a list of int, tuple of int or a int. "
                    f"Received max_size={max_size}"
                )
        elif not isinstance(max_size, int):
            raise ValueError(
                "`max_size` must a list of int, tuple of int or a int. "
                f"Received max_size={max_size}"
            )

        if isinstance(max_size, int):
            max_size = [max_size]
        self.max_size = sorted(max_size)
        self.interpolation = preprocessing_utils.get_interpolation(
            interpolation
        )
        self.antialias = antialias
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )

        # sample 1 max_size for the batch
        indices = self._random_generator.random_uniform(
            shape=(1,),
            minval=0,
            maxval=len(self.max_size),
            dtype=tf.int32,
        )
        max_sizes = tf.convert_to_tensor(self.max_size, dtype=tf.float32)
        max_sizes = tf.gather(max_sizes, indices[0])
        max_sizes = tf.reshape(max_sizes, shape=(1, 1))
        max_sizes = tf.tile(max_sizes, multiples=(batch_size, 1))

        larger_sides = tf.cast(
            tf.where(heights > widths, heights, widths), dtype=tf.float32
        )
        scales = max_sizes / larger_sides
        new_heights = tf.cast(tf.round(heights * scales), tf.int32)
        new_widths = tf.cast(tf.round(widths * scales), tf.int32)
        scaled_sizes = tf.concat((new_heights, new_widths), axis=-1)

        return {"scaled_sizes": scaled_sizes}

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        scaled_sizes = transformation["scaled_sizes"]
        transformation = {
            "scaled_sizes": tf.expand_dims(scaled_sizes, axis=0),
        }
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
                "`ResizeLongest()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`ResizeLongest(..., bounding_box_format='xyxy')`"
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
            images=images,
            source="rel_xyxy",
            target="xyxy",
        )
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
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
                "transformations": transformations,
            }
            segmentation_masks = tf.vectorized_map(
                self.augment_segmentation_mask_single,
                inputs,
            )
        else:
            scaled_size = transformations["scaled_sizes"]
            new_height = scaled_size[0][0]
            new_width = scaled_size[0][1]
            segmentation_masks = tf.image.resize(
                segmentation_masks,
                size=(new_height, new_width),
                method="nearest",
            )
            segmentation_masks = tf.cast(
                segmentation_masks, dtype=self.compute_dtype
            )
        return segmentation_masks

    def augment_segmentation_mask_single(self, inputs):
        segmentation_mask = inputs.get(
            augmentation_utils.SEGMENTATION_MASKS, None
        )
        transformation = inputs.get("transformations", None)
        # resize
        scaled_size = transformation["scaled_sizes"]
        new_height = scaled_size[0]
        new_width = scaled_size[1]
        segmentation_mask = tf.image.resize(
            segmentation_mask,
            size=(new_height, new_width),
            method="nearest",
        )
        return tf.cast(segmentation_mask, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_size": self.max_size,
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

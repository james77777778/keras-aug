import collections.abc

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class ResizeBySmallestSide(VectorizedBaseImageAugmentationLayer):
    """Resize images so that smallest side is equal to min_size, keeping the
    aspect ratio of the initial images.

    This layer produces outputs of the same min_size within a batch but may
    varying min_size across different batches if min_size is a list.

    Args:
        min_size: A list of int, tuple of int or a int. Represents smallest size
            of the image after the transformation. When using a list, max size
            will be randomly selected from the values in the list.
        interpolation: A string specifying the sampling method for
            resizing, defaults to "bilinear".
        antialias: A bool specifying whether to use antialias,
            defaults to False.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed: Used to create a random seed, defaults to None.
    """  # noqa: E501

    def __init__(
        self,
        min_size,
        interpolation="bilinear",
        antialias=False,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(min_size, collections.abc.Sequence):
            if not isinstance(min_size[0], int):
                raise ValueError(
                    "`min_size` must a list of int, tuple of int or a int. "
                    f"Received min_size={min_size}"
                )
        elif not isinstance(min_size, int):
            raise ValueError(
                "`min_size` must a list of int, tuple of int or a int. "
                f"Received min_size={min_size}"
            )

        if isinstance(min_size, int):
            min_size = [min_size]
        self.min_size = sorted(min_size)
        self.interpolation = preprocessing_utils.get_interpolation(
            interpolation
        )
        self.antialias = antialias
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )

        # sample 1 min_size for the batch
        indices = self._random_generator.random_uniform(
            shape=(1,),
            minval=0,
            maxval=len(self.min_size),
            dtype=tf.int32,
        )
        min_sizes = tf.convert_to_tensor(self.min_size, dtype=tf.float32)
        min_sizes = tf.gather(min_sizes, indices[0])
        min_sizes = tf.reshape(min_sizes, shape=(1, 1))
        min_sizes = tf.tile(min_sizes, multiples=(batch_size, 1))

        smaller_sides = tf.cast(
            tf.where(heights < widths, heights, widths), dtype=tf.float32
        )
        scales = min_sizes / smaller_sides
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
        inputs = {
            augmentation_utils.SEGMENTATION_MASKS: segmentation_masks,
            "transformations": transformations,
        }
        return tf.vectorized_map(
            self.augment_single_segmentation_mask,
            inputs,
        )

    def augment_single_segmentation_mask(self, inputs):
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
                "min_size": self.min_size,
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

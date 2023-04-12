import collections.abc

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentations._2d.keras_aug_2d_base_layer import (
    KerasAug2DBaseLayer,
)
from keras_aug.utils import augmentation_utils

H_AXIS = -3
W_AXIS = -2


@keras.utils.register_keras_serializable(package="keras_aug")
class ResizeLongest(KerasAug2DBaseLayer):
    """Resize images so that maximum side is equal to max_size, keeping the
    aspect ratio of the initial images.

    This layer produces outputs of the same max_size within a batch but varying
    max_size across different batches.

    Args:
        max_size: A list of int, tuple of int or a int. Represents maximum size
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
    """

    def __init__(
        self,
        max_size,
        interpolation="bilinear",
        antialias=False,
        seed=None,
        bounding_box_format=None,
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
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(images)
        heights = tf.cast(heights, dtype=tf.float32)
        widths = tf.cast(widths, dtype=tf.float32)

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

        return tf.image.resize(
            images,
            size=(new_height, new_width),
            method=self.interpolation,
            antialias=self.antialias,
        )

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_size": self.max_size,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

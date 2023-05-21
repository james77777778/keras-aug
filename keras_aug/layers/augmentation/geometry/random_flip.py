import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils import bounding_box as bounding_box_utils

# Defining modes for random flipping
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomFlip(VectorizedBaseRandomLayer):
    """Randomly flips the input images.

    This layer will flip the images horizontally and or vertically based on the
    ``mode`` attribute.

    Args:
        mode (str, optional): The flip mode to use. Supported values:
            ``"horizontal", "vertical", "horizontal_and_vertical"``. Defaults to
            ``"horizontal"``. ``"horizontal"`` is a left-right flip and
            ``"vertical"`` is a top-bottom flip.
        rate (float, optional): The frequency of flipping. ``1.0`` indicates
            that images are always flipped. ``0.0`` indicates no flipping.
            Defaults to ``0.5``.
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
        mode=HORIZONTAL,
        rate=0.5,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False
        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True
        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError(
                "RandomFlip layer {name} received an unknown mode="
                "{arg}".format(name=self.name, arg=mode)
            )
        self.mode = mode
        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
            )
        self.rate = rate
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        flip_horizontals = tf.zeros(shape=(batch_size, 1))
        flip_verticals = tf.zeros(shape=(batch_size, 1))
        if self.horizontal:
            flip_horizontals = self._random_generator.random_uniform(
                shape=(batch_size, 1)
            )
        if self.vertical:
            flip_verticals = self._random_generator.random_uniform(
                shape=(batch_size, 1)
            )
        return {
            "flip_horizontals": flip_horizontals,
            "flip_verticals": flip_verticals,
        }

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
        return self.flip_images(images, transformations)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations=None, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomFlip()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomFlip(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=raw_images,
        )
        boxes = bounding_boxes["boxes"]
        flip_horizontals = transformations["flip_horizontals"]
        flip_verticals = transformations["flip_verticals"]
        # broadcast
        flip_horizontals = flip_horizontals[:, tf.newaxis, :]
        flip_verticals = flip_verticals[:, tf.newaxis, :]
        if self.horizontal:
            boxes = tf.where(
                flip_horizontals > (1.0 - self.rate),
                self.flip_boxes_horizontal(boxes),
                boxes,
            )
        if self.vertical:
            boxes = tf.where(
                flip_verticals > (1.0 - self.rate),
                self.flip_boxes_vertical(boxes),
                boxes,
            )
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box_utils.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=raw_images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=raw_images,
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
        self, segmentation_masks, transformations=None, **kwargs
    ):
        return self.flip_images(segmentation_masks, transformations)

    def flip_images(self, images, transformations):
        original_shape = images.shape
        flip_horizontals = transformations["flip_horizontals"]
        flip_verticals = transformations["flip_verticals"]
        # broadcast
        flip_horizontals = flip_horizontals[:, tf.newaxis, tf.newaxis, :]
        flip_verticals = flip_verticals[:, tf.newaxis, tf.newaxis, :]
        if self.horizontal:
            images = tf.where(
                flip_horizontals > (1.0 - self.rate),
                tf.image.flip_left_right(images),
                images,
            )
        if self.vertical:
            images = tf.where(
                flip_verticals > (1.0 - self.rate),
                tf.image.flip_up_down(images),
                images,
            )
        images.set_shape(original_shape)
        return images

    def flip_boxes_horizontal(self, boxes):
        x1, x2, x3, x4 = tf.split(boxes, 4, axis=-1)
        outputs = tf.concat([1 - x3, x2, 1 - x1, x4], axis=-1)
        return outputs

    def flip_boxes_vertical(self, boxes):
        x1, x2, x3, x4 = tf.split(boxes, 4, axis=-1)
        outputs = tf.concat([x1, 1 - x4, x3, 1 - x2], axis=-1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "rate": self.rate,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

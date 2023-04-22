import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class CenterCrop(VectorizedBaseRandomLayer):
    """Center crops the images.

    CenterCrop crops the central portion of the images to a specified
    ``(height, width)``. If an image is smaller than the target size, it will be
    padded and then cropped.

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        position (str, optional): The padding method. Defaults to
            ``"center"``.
        padding_value (int|float, optional): The padding value.
            Defaults to ``0``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.
    """

    def __init__(
        self,
        height,
        width,
        position="center",
        padding_value=0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.height = height
        self.width = width
        self.position = augmentation_utils.get_padding_position(position)
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True
        self.force_output_dense_segmentation_masks = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(images)

        tops = tf.where(
            heights < self.height,
            tf.cast((self.height - heights) / 2, heights.dtype),
            tf.zeros_like(heights, dtype=heights.dtype),
        )
        bottoms = tf.where(
            heights < self.height,
            self.height - heights - tops,
            tf.zeros_like(heights, dtype=heights.dtype),
        )
        lefts = tf.where(
            widths < self.width,
            tf.cast((self.width - widths) / 2, widths.dtype),
            tf.zeros_like(widths, dtype=widths.dtype),
        )
        rights = tf.where(
            widths < self.width,
            self.width - widths - lefts,
            tf.zeros_like(widths, dtype=widths.dtype),
        )

        (tops, bottoms, lefts, rights) = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, self.position, self._random_generator
        )

        return {
            "pad_tops": tops,
            "pad_bottoms": bottoms,
            "pad_lefts": lefts,
            "pad_rights": rights,
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
        ori_height = tf.shape(images)[augmentation_utils.H_AXIS]
        ori_width = tf.shape(images)[augmentation_utils.W_AXIS]

        pad_top = transformations["pad_tops"][0][0]
        pad_bottom = transformations["pad_bottoms"][0][0]
        pad_left = transformations["pad_lefts"][0][0]
        pad_right = transformations["pad_rights"][0][0]
        paddings = tf.stack(
            (
                tf.zeros(shape=(2,), dtype=pad_top.dtype),
                tf.stack((pad_top, pad_bottom)),
                tf.stack((pad_left, pad_right)),
                tf.zeros(shape=(2,), dtype=pad_top.dtype),
            )
        )
        images = tf.pad(
            images, paddings=paddings, constant_values=self.padding_value
        )

        # center crop
        offset_height = (ori_height + pad_top + pad_bottom - self.height) // 2
        offset_width = (ori_width + pad_left + pad_right - self.width) // 2
        images = tf.image.crop_to_bounding_box(
            images,
            offset_height,
            offset_width,
            self.height,
            self.width,
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
                "`PadIfNeeded()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`PadIfNeeded(..., bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )

        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        pad_tops = tf.cast(transformations["pad_tops"], dtype=tf.float32)
        pad_lefts = tf.cast(transformations["pad_lefts"], dtype=tf.float32)
        pad_bottoms = tf.cast(transformations["pad_bottoms"], dtype=tf.float32)
        pad_rights = tf.cast(transformations["pad_rights"], dtype=tf.float32)
        heights, widths = augmentation_utils.get_images_shape(
            raw_images, dtype=tf.float32
        )
        offset_heights = (heights + pad_tops + pad_bottoms - self.height) // 2
        offset_widths = (widths + pad_lefts + pad_rights - self.width) // 2

        x1s += tf.expand_dims(pad_lefts - offset_widths, axis=1)
        y1s += tf.expand_dims(pad_tops - offset_heights, axis=1)
        x2s += tf.expand_dims(pad_lefts - offset_widths, axis=1)
        y2s += tf.expand_dims(pad_tops - offset_heights, axis=1)
        outputs = tf.concat([x1s, y1s, x2s, y2s], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
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
                "transformation": transformations,
            }
            segmentation_masks = tf.map_fn(
                self.augment_segmentation_mask_single,
                inputs,
                fn_output_signature=segmentation_masks.dtype,
            )
        else:
            pad_top = transformations["pad_tops"][0][0]
            pad_bottom = transformations["pad_bottoms"][0][0]
            pad_left = transformations["pad_lefts"][0][0]
            pad_right = transformations["pad_rights"][0][0]
            paddings = tf.stack(
                (
                    tf.zeros(shape=(2,), dtype=pad_top.dtype),
                    tf.stack((pad_top, pad_bottom)),
                    tf.stack((pad_left, pad_right)),
                    tf.zeros(shape=(2,), dtype=pad_top.dtype),
                )
            )
            segmentation_masks = tf.pad(
                segmentation_masks, paddings=paddings, constant_values=0
            )
        return segmentation_masks

    def augment_segmentation_mask_single(self, inputs):
        segmentation_mask = inputs.get(
            augmentation_utils.SEGMENTATION_MASKS, None
        )
        transformation = inputs.get("transformation", None)
        pad_top = transformation["pad_tops"][0]
        pad_bottom = transformation["pad_bottoms"][0]
        pad_left = transformation["pad_lefts"][0]
        pad_right = transformation["pad_rights"][0]
        paddings = tf.stack(
            (
                tf.stack((pad_top, pad_bottom)),
                tf.stack((pad_left, pad_right)),
                tf.zeros(shape=(2,), dtype=pad_top.dtype),
            )
        )
        segmentation_mask = tf.pad(
            segmentation_mask, paddings=paddings, constant_values=0
        )
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "position": self.position,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class ResizeAndPad(VectorizedBaseRandomLayer):
    """Resize and pad images to (`height`, `width`), keeping the aspect ratio
    of the initial images.

    Args:
        height: A integer specifying the height of result image.
        width: A integer specifying the width of result image.
        interpolation: Interpolation mode, defaults to `"bilinear"`. Supported
            values: `"nearest"`, `"bilinear"`.
        antialias: A bool specifying whether to use antialias,
            defaults to False.
        position: A string specifying the padding method, defaults
            to "center".
        padding_value: padding value, defaults to 0.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed: Used to create a random seed, defaults to None.
    """  # noqa: E501

    def __init__(
        self,
        height,
        width,
        interpolation="bilinear",
        antialias=False,
        position="center",
        padding_value=0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        super().__init__(force_output_dense_images=True, seed=seed, **kwargs)
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
        self.position = augmentation_utils.get_padding_position(position)
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # get scaled_sizes
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        scales = tf.minimum(self.height / heights, self.width / widths)
        new_heights = tf.cast(tf.round(heights * scales), tf.int32)
        new_widths = tf.cast(tf.round(widths * scales), tf.int32)
        scaled_sizes = tf.concat((new_heights, new_widths), axis=-1)

        # get padding values
        tops = tf.where(
            new_heights < self.height,
            tf.cast((self.height - new_heights) / 2, new_heights.dtype),
            tf.zeros_like(new_heights, dtype=new_heights.dtype),
        )
        bottoms = tf.where(
            new_heights < self.height,
            self.height - new_heights - tops,
            tf.zeros_like(new_heights, dtype=new_heights.dtype),
        )
        lefts = tf.where(
            new_widths < self.width,
            tf.cast((self.width - new_widths) / 2, new_widths.dtype),
            tf.zeros_like(new_widths, dtype=new_widths.dtype),
        )
        rights = tf.where(
            new_widths < self.width,
            self.width - new_widths - lefts,
            tf.zeros_like(new_widths, dtype=new_widths.dtype),
        )
        (tops, bottoms, lefts, rights) = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, self.position, self._random_generator
        )

        return {
            "scaled_sizes": scaled_sizes,
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
        # resize
        # currently, only support same size within dense batch
        # this layer might produce varying size with ragged batch
        scaled_sizes = transformations["scaled_sizes"]
        new_height = scaled_sizes[0][0]
        new_width = scaled_sizes[0][1]
        images = tf.image.resize(
            images,
            size=(new_height, new_width),
            method=self.interpolation,
            antialias=self.antialias,
        )
        # pad
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
                "`ResizeAndPad()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`ResizeAndPad(..., bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )

        # get new/old ratios
        ori_heights, ori_widths = augmentation_utils.get_images_shape(
            raw_images, dtype=tf.float32
        )
        scaled_sizes = transformations["scaled_sizes"]
        new_heights = tf.cast(scaled_sizes[..., 0:1], dtype=tf.float32)
        new_widths = tf.cast(scaled_sizes[..., 1:2], dtype=tf.float32)
        height_ratios = new_heights / ori_heights
        widths_ratios = new_widths / ori_widths

        pad_tops = tf.cast(transformations["pad_tops"], dtype=tf.float32)
        pad_lefts = tf.cast(transformations["pad_lefts"], dtype=tf.float32)
        # broadcast
        height_ratios = height_ratios[..., tf.newaxis]
        widths_ratios = widths_ratios[..., tf.newaxis]
        pad_tops = pad_tops[..., tf.newaxis]
        pad_lefts = pad_lefts[..., tf.newaxis]

        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        x1s = x1s * widths_ratios + pad_lefts
        x2s = x2s * widths_ratios + pad_lefts
        y1s = y1s * height_ratios + pad_tops
        y2s = y2s * height_ratios + pad_tops

        boxes = tf.concat([x1s, y1s, x2s, y2s], axis=-1)
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
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
            # resize
            scaled_size = transformations["scaled_sizes"]
            new_height = scaled_size[0][0]
            new_width = scaled_size[0][1]
            segmentation_masks = tf.image.resize(
                segmentation_masks,
                size=(new_height, new_width),
                method="nearest",
                antialias=self.antialias,
            )
            # pad
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
                segmentation_masks,
                paddings=paddings,
                constant_values=0,
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
            antialias=self.antialias,
        )
        # pad
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
            segmentation_mask,
            paddings=paddings,
            constant_values=0,
        )
        return tf.cast(segmentation_mask, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
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

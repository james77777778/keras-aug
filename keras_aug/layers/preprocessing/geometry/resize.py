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

    Resize will resize the images to ``(height, width)``. Set
    ``crop_to_aspect_ratio`` or ``pad_to_aspect_ratio`` to ``True`` to keep
    the aspect ratio.

    When ``crop_to_aspect_ratio`` or ``pad_to_aspect_ratio`` is set to ``True``.
    You can control the cropping position or padding position by setting
    ``position``.

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        interpolation (str, optional): The interpolation mode.
            Supported values: ``"nearest", "bilinear"``.
            Defaults to `"bilinear"`.
        antialias (bool, optional): Whether to use antialias. Defaults to
            ``False``.
        crop_to_aspect_ratio (bool, optional): If ``True``, the output images
            will be cropped to return the largest possible window in the images.
            Defaults to ``False``.
        pad_to_aspect_ratio (bool, optional): If ``True``, the output images
            will be padded to return the largest possible resize of the images.
            Defaults to ``False``.
        position (str, optional): The padding method.
            Supported values: ``"center", "top_left", "top_right", "bottom_left", "bottom_right", "random"``.
            Defaults to ``"center"``.
        padding_value (int|float, optional): The padding value.
            Defaults to ``0``.
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
        antialias=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        position="center",
        padding_value=0,
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
        self.position = augmentation_utils.get_padding_position(position)
        self.padding_value = padding_value
        if crop_to_aspect_ratio is True and pad_to_aspect_ratio is True:
            raise ValueError(
                "Resize expects at most one of ``crop_to_aspect_ratio`` or "
                "``pad_to_aspect_ratio`` to be ``True``"
            )
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True
        self.force_output_dense_segmentation_masks = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        if not self.crop_to_aspect_ratio and not self.pad_to_aspect_ratio:
            return {"dummy": tf.zeros((batch_size,))}

        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        # get scaled_sizes
        if self.crop_to_aspect_ratio:
            scales = tf.maximum(self.height / heights, self.width / widths)
        elif self.pad_to_aspect_ratio:
            scales = tf.minimum(self.height / heights, self.width / widths)
        new_heights = tf.cast(tf.round(heights * scales), tf.int32)
        new_widths = tf.cast(tf.round(widths * scales), tf.int32)
        scaled_sizes = tf.concat((new_heights, new_widths), axis=-1)

        # get padding values
        if self.crop_to_aspect_ratio:
            tops = tf.where(
                new_heights > self.height,
                tf.cast((new_heights - self.height) / 2, tf.int32),
                0,
            )
            bottoms = tf.where(
                new_heights > self.height, new_heights - self.height - tops, 0
            )
            lefts = tf.where(
                new_widths > self.width,
                tf.cast((new_widths - self.width) / 2, tf.int32),
                0,
            )
            rights = tf.where(
                new_widths > self.width, new_widths - self.width - lefts, 0
            )
        elif self.pad_to_aspect_ratio:
            tops = tf.where(
                new_heights < self.height,
                tf.cast((self.height - new_heights) / 2, tf.int32),
                0,
            )
            bottoms = tf.where(
                new_heights < self.height, self.height - new_heights - tops, 0
            )
            lefts = tf.where(
                new_widths < self.width,
                tf.cast((self.width - new_widths) / 2, tf.int32),
                0,
            )
            rights = tf.where(
                new_widths < self.width, self.width - new_widths - lefts, 0
            )
        (tops, bottoms, lefts, rights) = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, self.position, self._random_generator
        )
        return {
            "scaled_sizes": scaled_sizes,
            "tops": tops,
            "bottoms": bottoms,
            "lefts": lefts,
            "rights": rights,
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
        if not self.crop_to_aspect_ratio and not self.pad_to_aspect_ratio:
            images = tf.image.resize(
                images,
                size=(self.height, self.width),
                method=self.interpolation,
                antialias=self.antialias,
            )
            return tf.cast(images, dtype=self.compute_dtype)

        # resize keeping aspect ratio
        if self.crop_to_aspect_ratio:
            images = self.resize_with_crop_to_aspect_ratio(
                images, transformations, self.interpolation
            )
        elif self.pad_to_aspect_ratio:
            images = self.resize_with_pad_to_aspect_ratio(
                images, transformations, self.interpolation, self.padding_value
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
        if not self.crop_to_aspect_ratio and not self.pad_to_aspect_ratio:
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

        # resize keeping aspect ratio
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
        tops = tf.cast(transformations["tops"], dtype=tf.float32)
        lefts = tf.cast(transformations["lefts"], dtype=tf.float32)
        # broadcast
        height_ratios = height_ratios[..., tf.newaxis]
        widths_ratios = widths_ratios[..., tf.newaxis]
        tops = tops[..., tf.newaxis]
        lefts = lefts[..., tf.newaxis]

        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        if self.crop_to_aspect_ratio:
            x1s = x1s * widths_ratios - lefts
            x2s = x2s * widths_ratios - lefts
            y1s = y1s * height_ratios - tops
            y2s = y2s * height_ratios - tops
        elif self.pad_to_aspect_ratio:
            x1s = x1s * widths_ratios + lefts
            x2s = x2s * widths_ratios + lefts
            y1s = y1s * height_ratios + tops
            y2s = y2s * height_ratios + tops
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
            return tf.cast(segmentation_masks, dtype=self.compute_dtype)
        # resize
        if not self.crop_to_aspect_ratio and not self.pad_to_aspect_ratio:
            segmentation_masks = tf.image.resize(
                segmentation_masks,
                size=(self.height, self.width),
                method="nearest",
            )

        # resize keeping aspect ratio
        if self.crop_to_aspect_ratio:
            segmentation_masks = self.resize_with_crop_to_aspect_ratio(
                segmentation_masks, transformations, "nearest"
            )
        elif self.pad_to_aspect_ratio:
            segmentation_masks = self.resize_with_pad_to_aspect_ratio(
                segmentation_masks, transformations, "nearest", 0
            )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

    def augment_segmentation_mask_single(self, inputs):
        segmentation_mask = inputs.get(
            augmentation_utils.SEGMENTATION_MASKS, None
        )
        transformation = inputs.get("transformations", None)
        # resize
        if not self.crop_to_aspect_ratio and not self.pad_to_aspect_ratio:
            return tf.image.resize(
                segmentation_mask,
                size=(self.height, self.width),
                method="nearest",
            )

        # resize keeping aspect ratio
        scaled_size = transformation["scaled_sizes"]
        new_height = scaled_size[0]
        new_width = scaled_size[1]
        segmentation_mask = tf.image.resize(
            segmentation_mask,
            size=(new_height, new_width),
            method="nearest",
        )
        if self.crop_to_aspect_ratio:
            # crop
            top = tf.cast(transformation["tops"][0], dtype=tf.float32)
            left = tf.cast(transformation["lefts"][0], dtype=tf.float32)
            segmentation_mask = tf.image.crop_to_bounding_box(
                segmentation_mask, top, left, self.height, self.width
            )
        elif self.pad_to_aspect_ratio:
            # pad
            pad_top = transformation["tops"][0]
            pad_bottom = transformation["bottoms"][0]
            pad_left = transformation["lefts"][0]
            pad_right = transformation["rights"][0]
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

        return segmentation_mask

    def resize_with_crop_to_aspect_ratio(self, images, transformations, method):
        batch_size = tf.shape(images)[0]
        scaled_sizes = transformations["scaled_sizes"]
        new_height = scaled_sizes[0][0]
        new_width = scaled_sizes[0][1]
        # resize
        images = tf.image.resize(
            images,
            size=(new_height, new_width),
            method=method,
            antialias=self.antialias,
        )
        # crop
        y1s = tf.cast(transformations["tops"], dtype=tf.float32)
        y2s = tf.cast(y1s + self.height, dtype=tf.float32)
        x1s = tf.cast(transformations["lefts"], dtype=tf.float32)
        x2s = tf.cast(x1s + self.width, dtype=tf.float32)
        # normalize boxes
        new_height = tf.cast(new_height, dtype=tf.float32)
        new_width = tf.cast(new_width, dtype=tf.float32)
        y1s /= new_height
        y2s /= new_height
        x1s /= new_width
        x2s /= new_width
        boxes = tf.concat([y1s, x1s, y2s, x2s], axis=-1)
        images = tf.image.crop_and_resize(
            images,
            boxes,
            tf.range(batch_size),
            [self.height, self.width],
            method="nearest",
        )
        return images

    def resize_with_pad_to_aspect_ratio(
        self, images, transformations, method, fill_value
    ):
        # resize
        scaled_sizes = transformations["scaled_sizes"]
        new_height = scaled_sizes[0][0]
        new_width = scaled_sizes[0][1]
        images = tf.image.resize(
            images,
            size=(new_height, new_width),
            method=method,
            antialias=self.antialias,
        )
        # pad
        pad_top = transformations["tops"][0][0]
        pad_bottom = transformations["bottoms"][0][0]
        pad_left = transformations["lefts"][0][0]
        pad_right = transformations["rights"][0][0]
        paddings = tf.stack(
            (
                tf.zeros(shape=(2,), dtype=tf.int32),
                tf.stack((pad_top, pad_bottom)),
                tf.stack((pad_left, pad_right)),
                tf.zeros(shape=(2,), dtype=tf.int32),
            )
        )
        images = tf.pad(images, paddings=paddings, constant_values=fill_value)
        return images

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
                "pad_to_aspect_ratio": self.pad_to_aspect_ratio,
                "position": self.position,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

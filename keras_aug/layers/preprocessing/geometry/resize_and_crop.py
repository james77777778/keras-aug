import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class ResizeAndCrop(VectorizedBaseRandomLayer):
    """Resizes and crops the images while keeping the aspect ratio.

    ResizeAndCrop will firstly resize the images to fit in ``(height, width)``
    by smaller side while keeping the aspect ratio of the initial images and
    then crop to exactly ``(height, width)``.

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        interpolation (str, optional): The interpolation mode.
            Supported values: ``"nearest", "bilinear"``.
            Defaults to `"bilinear"`.
        antialias (bool, optional): Whether to use antialias. Defaults to
            ``False``.
        position (str, optional): The cropping method.
            Supported values: ``"center", "top_left", "top_right", "bottom_left", "bottom_right", "random"``.
            Defaults to ``"center"``.
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
        position="center",
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
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True
        self.force_output_dense_segmentation_masks = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # get scaled_sizes
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        scales = tf.maximum(self.height / heights, self.width / widths)
        new_heights = tf.cast(tf.round(heights * scales), tf.int32)
        new_widths = tf.cast(tf.round(widths * scales), tf.int32)
        scaled_sizes = tf.concat((new_heights, new_widths), axis=-1)

        # get cropping values
        tops = tf.where(
            new_heights > self.height,
            tf.cast((new_heights - self.height) / 2, new_heights.dtype),
            tf.zeros_like(self.height, dtype=new_heights.dtype),
        )
        bottoms = tf.where(
            new_heights > self.height,
            new_heights - self.height - tops,
            tf.zeros_like(self.height, dtype=new_heights.dtype),
        )
        lefts = tf.where(
            new_widths > self.width,
            tf.cast((new_widths - self.width) / 2, new_widths.dtype),
            tf.zeros_like(self.width, dtype=new_widths.dtype),
        )
        rights = tf.where(
            new_widths > self.width,
            new_widths - self.width - lefts,
            tf.zeros_like(self.width, dtype=new_widths.dtype),
        )
        (tops, _, lefts, _) = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, self.position, self._random_generator
        )

        return {
            "scaled_sizes": scaled_sizes,
            "crop_tops": tops,
            "crop_lefts": lefts,
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
        batch_size = tf.shape(images)[0]
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
        # crop
        y1s = tf.cast(transformations["crop_tops"], dtype=tf.float32)
        y2s = tf.cast(y1s + self.height, dtype=tf.float32)
        x1s = tf.cast(transformations["crop_lefts"], dtype=tf.float32)
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
                "`ResizeAndCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`ResizeAndCrop(..., bounding_box_format='xyxy')`"
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

        crop_tops = tf.cast(transformations["crop_tops"], dtype=tf.float32)
        crop_lefts = tf.cast(transformations["crop_lefts"], dtype=tf.float32)
        # broadcast
        height_ratios = height_ratios[..., tf.newaxis]
        widths_ratios = widths_ratios[..., tf.newaxis]
        crop_tops = crop_tops[..., tf.newaxis]
        crop_lefts = crop_lefts[..., tf.newaxis]

        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        x1s = x1s * widths_ratios - crop_lefts
        x2s = x2s * widths_ratios - crop_lefts
        y1s = y1s * height_ratios - crop_tops
        y2s = y2s * height_ratios - crop_tops

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
            batch_size = tf.shape(segmentation_masks)[0]
            # resize
            scaled_size = transformations["scaled_sizes"]
            new_height = scaled_size[0][0]
            new_width = scaled_size[0][1]
            segmentation_masks = tf.image.resize(
                segmentation_masks,
                size=(new_height, new_width),
                method="nearest",
            )
            # crop
            new_height = tf.cast(new_height, dtype=tf.float32)
            new_width = tf.cast(new_width, dtype=tf.float32)
            y1s = tf.cast(transformations["crop_tops"], dtype=tf.float32)
            y2s = tf.cast(y1s + self.height, dtype=tf.float32)
            x1s = tf.cast(transformations["crop_lefts"], dtype=tf.float32)
            x2s = tf.cast(x1s + self.width, dtype=tf.float32)
            y1s /= new_height
            y2s /= new_height
            x1s /= new_width
            x2s /= new_width
            boxes = tf.concat([y1s, x1s, y2s, x2s], axis=-1)
            segmentation_masks = tf.image.crop_and_resize(
                segmentation_masks,
                boxes,
                tf.range(batch_size),
                [self.height, self.width],
                method="nearest",
            )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

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
        # crop
        crop_top = tf.cast(transformation["crop_tops"][0], dtype=tf.float32)
        crop_left = tf.cast(transformation["crop_lefts"][0], dtype=tf.float32)
        segmentation_mask = tf.image.crop_to_bounding_box(
            segmentation_mask, crop_top, crop_left, self.height, self.width
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
                "position": self.position,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

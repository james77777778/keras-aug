import math

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentations._2d.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomAffine(VectorizedBaseRandomLayer):
    """A preprocessing layer which randomly affines transformation of the images
    keeping center invariant.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        rotation_factor: a float represented as fraction of 2 Pi, or a tuple of
            size 2 representing lower and upper bound for rotating clockwise and
            counter-clockwise. A positive values means rotating counter
            clock-wise, while a negative value means clock-wise. When
            represented as a single float, this value is used for both the upper
            and lower bound. For instance, `factor=(-0.2, 0.3)` results in an
            output rotation by a random amount in the range
            `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in an output
            rotating by a random amount in the range `[-20% * 2pi, 20% * 2pi]`.
        translation_height_factor: a float represented as fraction of value, or
            a tuple of size 2 representing lower and upper bound for shifting
            vertically. A negative value means shifting image up, while a
            positive value means shifting image down. When represented as a
            single positive float, this value is used for both the upper and
            lower bound. For instance, `height_factor=(-0.2, 0.3)` results in an
            output shifted by a random amount in the range `[-20%, +30%]`.
            `height_factor=0.2` results in an output height shifted by a random
            amount in the range `[-20%, +20%]`.
        translation_width_factor: a float represented as fraction of value, or a
            tuple of size 2 representing lower and upper bound for shifting
            horizontally. A negative value means shifting image left, while a
            positive value means shifting image right. When represented as a
            single positive float, this value is used for both the upper and
            lower bound. For instance, `width_factor=(-0.2, 0.3)` results in an
            output shifted left by 20%, and shifted right by 30%.
            `width_factor=0.2` results in an output height shifted left or right
            by 20%.
        zoom_height_factor: a float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound for zooming vertically.
            When represented as a single float, this value is used for both the
            upper and lower bound. A positive value means zooming out, while a
            negative value means zooming in. For instance,
            `height_factor=(0.2, 0.3)` result in an output zoomed out by a
            random amount in the range `[+20%, +30%]`.
            `height_factor=(-0.3, -0.2)` result in an output zoomed in by a
            random amount in the range `[-30%, -20%]`.
        zoom_width_factor: a float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound for zooming
            horizontally. When represented as a single float, this value is used
            for both the upper and lower bound. For instance,
            `width_factor=(0.2, 0.3)` result in an output zooming out between
            20% to 30%. `width_factor=(-0.3, -0.2)` result in an output zooming
            in between 20% to 30%.
        shear_height_factor: a float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound for shearing
            vertically. When represented as a single float, this value is used
            for both the upper and lower bound. For instance,
            `width_factor=(0.2, 0.3)` result in an output shearing between
            20% to 30%. `width_factor=(-0.3, -0.2)` result in an output shearing
            between 20% to 30%.
        shear_width_factor: a float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound for shearing
            horizontally. When represented as a single float, this value is used
            for both the upper and lower bound. For instance,
            `width_factor=(0.2, 0.3)` result in an output zooming out between
            20% to 30%. `width_factor=(-0.3, -0.2)` result in an output shearing
            between 20% to 30%.
        interpolation: Interpolation mode, defaults to `"bilinear"`. Supported
            values: `"nearest"`, `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`), defaults to
            `"constant"`.
            - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended
            by reflecting about the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)` The input is extended
            by filling all values beyond the edge with the same constant value
            k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
            wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended
            by the nearest pixel.
        fill_value: a float represents the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed: Integer. Used to create a random seed.
    """  # noqa: E501

    def __init__(
        self,
        rotation_factor,
        translation_height_factor,
        translation_width_factor,
        zoom_height_factor,
        zoom_width_factor,
        shear_height_factor,
        shear_width_factor,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        # rotation
        if isinstance(rotation_factor, (tuple, list)):
            lower = rotation_factor[0] * 2.0 * math.pi
            upper = rotation_factor[1] * 2.0 * math.pi
        else:
            lower = -rotation_factor * 2.0 * math.pi
            upper = rotation_factor * 2.0 * math.pi
        self.rotation_factor_input = rotation_factor
        self.rotation_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-2.0 * math.pi, max_value=2.0 * math.pi
        )
        # translation
        if isinstance(translation_height_factor, (tuple, list)):
            lower = translation_height_factor[0]
            upper = translation_height_factor[1]
        else:
            lower = -translation_height_factor
            upper = translation_height_factor
        self.translation_height_factor_input = translation_height_factor
        self.translation_height_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )
        if isinstance(translation_width_factor, (tuple, list)):
            lower = translation_width_factor[0]
            upper = translation_width_factor[1]
        else:
            lower = -translation_width_factor
            upper = translation_width_factor
        self.translation_width_factor_input = translation_width_factor
        self.translation_width_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )
        # zoom
        if isinstance(zoom_height_factor, (tuple, list)):
            lower = 1.0 + zoom_height_factor[0]
            upper = 1.0 + zoom_height_factor[1]
        else:
            lower = 1.0 - zoom_height_factor
            upper = 1.0 + zoom_height_factor
        self.zoom_height_factor_input = zoom_height_factor
        self.zoom_height_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=0, max_value=None
        )
        if isinstance(zoom_width_factor, (tuple, list)):
            lower = 1.0 + zoom_width_factor[0]
            upper = 1.0 + zoom_width_factor[1]
        else:
            lower = 1.0 - zoom_width_factor
            upper = 1.0 + zoom_width_factor
        self.zoom_width_factor_input = zoom_width_factor
        self.zoom_width_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=0, max_value=None
        )
        # shear
        if isinstance(shear_height_factor, (tuple, list)):
            lower = shear_height_factor[0]
            upper = shear_height_factor[1]
        else:
            lower = -shear_height_factor
            upper = shear_height_factor
        self.shear_height_factor_input = shear_height_factor
        self.shear_height_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )
        if isinstance(shear_width_factor, (tuple, list)):
            lower = shear_width_factor[0]
            upper = shear_width_factor[1]
        else:
            lower = -shear_width_factor
            upper = shear_width_factor
        self.shear_width_factor_input = shear_width_factor
        self.shear_width_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )

        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )

        angles = self.rotation_factor(shape=(batch_size, 1))
        translation_heights = self.translation_height_factor(
            shape=(batch_size, 1)
        )
        translation_widths = self.translation_width_factor(
            shape=(batch_size, 1)
        )
        translations = tf.concat(
            [translation_widths, translation_heights], axis=1
        )
        zoom_heights = self.zoom_height_factor(shape=(batch_size, 1))
        zoom_widths = self.zoom_width_factor(shape=(batch_size, 1))
        zooms = tf.concat([zoom_widths, zoom_heights], axis=1)
        shear_heights = self.shear_height_factor(shape=(batch_size, 1))
        shear_widths = self.shear_width_factor(shape=(batch_size, 1))
        shears = tf.concat([shear_widths, shear_heights], axis=1)

        # combine matrix
        rotation_matrixes = augmentation_utils.get_rotation_matrix(
            angles, heights, widths, to_square=True
        )
        translation_matrixes = augmentation_utils.get_translation_matrix(
            translations, heights, widths, to_square=True
        )
        zoom_matrixes = augmentation_utils.get_zoom_matrix(
            zooms, heights, widths, to_square=True
        )
        shear_matrixes = augmentation_utils.get_shear_matrix(
            shears, to_square=True
        )
        # (batch_size, 3, 3)
        combined_matrixes = (
            translation_matrixes
            @ shear_matrixes
            @ zoom_matrixes
            @ rotation_matrixes
        )

        return {
            "angles": angles,
            "translations": translations,
            "zooms": zooms,
            "shears": shears,
            "combined_matrixes": combined_matrixes,
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
        combined_matrixes = transformations["combined_matrixes"]
        combined_matrixes = tf.reshape(
            combined_matrixes, shape=(batch_size, -1)
        )
        combined_matrixes = combined_matrixes[:, :-1]

        images = preprocessing_utils.transform(
            images,
            combined_matrixes,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomAffine()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomAffine(bounding_box_format='xyxy')`"
            )
        heights, widths = augmentation_utils.get_images_shape(
            raw_images, dtype=self.compute_dtype
        )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
            dtype=tf.float32,
        )

        # process rotations
        origin_x = widths / 2
        origin_y = heights / 2
        angles = -transformations["angles"]
        angles = angles[:, tf.newaxis, tf.newaxis]
        boxes = bounding_boxes["boxes"]
        # points: (batch_size, max_num_boxes, 4, 2)
        points = tf.stack(
            [
                tf.stack([boxes[:, :, 0], boxes[:, :, 1]], axis=2),
                tf.stack([boxes[:, :, 2], boxes[:, :, 1]], axis=2),
                tf.stack([boxes[:, :, 2], boxes[:, :, 3]], axis=2),
                tf.stack([boxes[:, :, 0], boxes[:, :, 3]], axis=2),
            ],
            axis=2,
        )
        point_x_offsets = points[..., 0] - origin_x[..., tf.newaxis]
        point_y_offsets = points[..., 1] - origin_y[..., tf.newaxis]
        new_x = (
            origin_x[..., tf.newaxis, tf.newaxis]
            + tf.multiply(tf.cos(angles), point_x_offsets[..., tf.newaxis])
            - tf.multiply(tf.sin(angles), point_y_offsets[..., tf.newaxis])
        )
        new_y = (
            origin_y[..., tf.newaxis, tf.newaxis]
            + tf.multiply(tf.sin(angles), point_x_offsets[..., tf.newaxis])
            + tf.multiply(tf.cos(angles), point_y_offsets[..., tf.newaxis])
        )
        out = tf.concat([new_x, new_y], axis=3)
        min_cordinates = tf.math.reduce_min(out, axis=2)
        max_cordinates = tf.math.reduce_max(out, axis=2)
        boxes = tf.concat([min_cordinates, max_cordinates], axis=2)

        # process translations
        translations = transformations["translations"]
        translation_widths = translations[:, 0:1] * widths
        translation_heights = translations[:, 1:2] * heights
        _x1s = boxes[:, :, 0] + translation_widths
        _y1s = boxes[:, :, 1] + translation_heights
        _x2s = boxes[:, :, 2] + translation_widths
        _y2s = boxes[:, :, 3] + translation_heights

        # process shear
        shears = transformations["shears"]
        shear_widths = shears[:, 0:1]
        shear_heights = shears[:, 1:2]
        # x1, x2
        x1_tops = _x1s - (shear_widths * _y1s)
        x1_bottoms = _x1s - (shear_widths * _y2s)
        x1s = tf.where(shear_widths < 0, x1_tops, x1_bottoms)
        x2_tops = _x2s - (shear_widths * _y1s)
        x2_bottoms = _x2s - (shear_widths * _y2s)
        x2s = tf.where(shear_widths < 0, x2_bottoms, x2_tops)
        # y1, y2
        y1_lefts = _y1s - (shear_heights * _x1s)
        y1_rights = _y1s - (shear_heights * _x2s)
        y1s = tf.where(shear_heights > 0, y1_rights, y1_lefts)
        y2_lefts = _y2s - (shear_heights * _x1s)
        y2_rights = _y2s - (shear_heights * _x2s)
        y2s = tf.where(shear_heights > 0, y2_lefts, y2_rights)

        # process zoom
        zooms = transformations["zooms"]
        zoom_widths = zooms[:, 0:1]
        zoom_heights = zooms[:, 1:2]
        x_offsets = ((widths - 1.0) / 2.0) * (1.0 - zoom_widths)
        y_offsets = ((heights - 1.0) / 2.0) * (1.0 - zoom_heights)
        x1s = (x1s - x_offsets) / zoom_widths
        x2s = (x2s - x_offsets) / zoom_widths
        y1s = (y1s - y_offsets) / zoom_heights
        y2s = (y2s - y_offsets) / zoom_heights

        boxes = tf.stack([x1s, y1s, x2s, y2s], axis=-1)
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=raw_images,
        )
        # coordinates cannot be float values, it is cast to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=raw_images,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, raw_images=None, **kwargs
    ):
        batch_size = tf.shape(raw_images)[0]
        combined_matrixes = transformations["combined_matrixes"]
        combined_matrixes = tf.reshape(
            combined_matrixes, shape=(batch_size, -1)
        )
        combined_matrixes = combined_matrixes[:, :-1]

        segmentation_masks = preprocessing_utils.transform(
            segmentation_masks,
            combined_matrixes,
            fill_mode=self.fill_mode,
            fill_value=0,
            interpolation="nearest",
        )
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor_input,
                "translation_height_factor": self.translation_height_factor_input,  # noqa: E501
                "translation_width_factor": self.translation_width_factor_input,
                "zoom_height_factor": self.zoom_height_factor_input,
                "zoom_width_factor": self.zoom_width_factor_input,
                "shear_height_factor": self.shear_height_factor_input,
                "shear_width_factor": self.shear_width_factor_input,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

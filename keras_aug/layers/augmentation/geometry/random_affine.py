import math

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils import bounding_box as bounding_box_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomAffine(VectorizedBaseRandomLayer):
    """Randomly affines transformation of the images keeping center invariant.

    Randomly affines by rotation, translation, zoom and shear. RandomAffine
    processes the images by combined transformation matrix, so it is fast.

    Args:
        rotation_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range of the degree for random rotation. When represented as a
            single float, the factor will be picked between
            ``[0.0 - lower, 0.0 + upper]``. A positive value means rotating
            counter clock-wise, while a negative value means clock-wise.
            Defaults to ``None``.
        translation_height_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range for random vertical translation. When represented as a single
            float, the factor will be picked between
            ``[0.0 - lower, 0.0 + upper]``. A negative value means shifting image
            up, while a positive value means shifting image down.
            Defaults to ``None``.
        translation_width_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range for random horizontal translation. When represented as a
            single float, the factor will be picked between
            ``[0.0 - lower, 0.0 + upper]``. A negative value means shifting
            image left, while a positive value means shifting image right.
            Defaults to ``None``.
        zoom_height_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range for random vertical zoom. When represented as a
            single float, the factor will be picked between
            ``[1.0 - lower, 1.0 + upper]``. A negative value means zooming in
            while a positive value means zooming out. Defaults to ``None``.
        zoom_width_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range for random horizontal zoom. When represented as a
            single float, the factor will be picked between
            ``[1.0 - lower, 1.0 + upper]``. A negative value means zooming in
            while a positive value means zooming out. Defaults to ``None``.
        shear_height_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range for random vertical shear. When represented as a
            single float, the factor will be picked between
            ``[0.0 - lower, 0.0 + upper]``. Defaults to ``None``.
        shear_width_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range for random horizontal shear. When represented as a
            single float, the factor will be picked between
            ``[0.0 - lower, 0.0 + upper]``. Defaults to ``None``.
        same_zoom_factor (bool, optional): If True, the zoom factor sampled from
            ``zoom_height_factor`` will be applied to both height and width.
            It is useful to keep aspect ratio. Defaults to ``False``.
        interpolation (str, optional): The interpolation mode. Supported values:
            ``"nearest", "bilinear"``. Defaults to `"bilinear"`.
        fill_mode (str, optional): The fill mode. Supported values:
            ``"constant", "reflect", "wrap", "nearest"``. Defaults to
            ``"constant"``.
        fill_value (int|float, optional): The value to be filled outside the
            boundaries when ``fill_mode="constant"``. Defaults to ``0``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        bounding_box_min_area_ratio (float, optional): The threshold to
            apply sanitize_bounding_boxes. Defaults to ``None``.
        bounding_box_max_aspect_ratio (float, optional): The threshold to
            apply sanitize_bounding_boxes. Defaults to ``None``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        rotation_factor=None,
        translation_height_factor=None,
        translation_width_factor=None,
        zoom_height_factor=None,
        zoom_width_factor=None,
        shear_height_factor=None,
        shear_width_factor=None,
        same_zoom_factor=False,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        bounding_box_format=None,
        bounding_box_min_area_ratio=None,
        bounding_box_max_aspect_ratio=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.rotation_factor = None
        self.translation_height_factor = None
        self.translation_width_factor = None
        self.zoom_height_factor = None
        self.zoom_width_factor = None
        self.shear_height_factor = None
        self.shear_width_factor = None
        # rotation
        if rotation_factor is not None:
            self.rotation_factor = augmentation_utils.parse_factor(
                rotation_factor,
                min_value=-180,
                max_value=180,
                center_value=0,
                seed=seed,
            )
        # translation
        if translation_height_factor is not None:
            self.translation_height_factor = augmentation_utils.parse_factor(
                translation_height_factor,
                min_value=-1,
                max_value=1,
                center_value=0.0,
                seed=seed,
            )
        if translation_width_factor is not None:
            self.translation_width_factor = augmentation_utils.parse_factor(
                translation_width_factor,
                min_value=-1,
                max_value=1,
                center_value=0.0,
                seed=seed,
            )
        # zoom
        if zoom_height_factor is not None:
            self.zoom_height_factor = augmentation_utils.parse_factor(
                zoom_height_factor,
                min_value=0,
                max_value=None,
                center_value=1.0,
                seed=seed,
            )
        if zoom_width_factor is not None:
            self.zoom_width_factor = augmentation_utils.parse_factor(
                zoom_width_factor,
                min_value=0,
                max_value=None,
                center_value=1.0,
                seed=seed,
            )
        # shear
        if shear_height_factor is not None:
            self.shear_height_factor = augmentation_utils.parse_factor(
                shear_height_factor,
                min_value=-1,
                max_value=1,
                center_value=0.0,
                seed=seed,
            )
        if shear_width_factor is not None:
            self.shear_width_factor = augmentation_utils.parse_factor(
                shear_width_factor,
                min_value=-1,
                max_value=1,
                center_value=0.0,
                seed=seed,
            )

        self.same_zoom_factor = same_zoom_factor

        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.bounding_box_format = bounding_box_format
        self.bounding_box_min_area_ratio = bounding_box_min_area_ratio
        self.bounding_box_max_aspect_ratio = bounding_box_max_aspect_ratio
        self.seed = seed

        # decide whether to enable the augmentation
        self._enable_rotation = augmentation_utils.is_factor_working(
            self.rotation_factor, not_working_value=0.0
        )
        _enable_translation_height = augmentation_utils.is_factor_working(
            self.translation_height_factor, not_working_value=0.0
        )
        _enable_translation_width = augmentation_utils.is_factor_working(
            self.translation_width_factor, not_working_value=0.0
        )
        self._enable_translation = (
            _enable_translation_height or _enable_translation_width
        )
        _enable_zoom_height = augmentation_utils.is_factor_working(
            self.zoom_height_factor, not_working_value=0.0
        )
        _enable_zoom_width = augmentation_utils.is_factor_working(
            self.zoom_width_factor, not_working_value=0.0
        )
        self._enable_zoom = _enable_zoom_height or _enable_zoom_width
        _enable_shear_height = augmentation_utils.is_factor_working(
            self.shear_height_factor, not_working_value=0.0
        )
        _enable_shear_width = augmentation_utils.is_factor_working(
            self.shear_width_factor, not_working_value=0.0
        )
        self._enable_shear = _enable_shear_height or _enable_shear_width

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # cast to float32 to avoid numerical issue
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        factor_shape = (batch_size, 1)
        # dummy
        angles = tf.zeros(factor_shape, dtype=tf.float32)
        translation_heights = tf.zeros(factor_shape, dtype=tf.float32)
        translation_widths = tf.zeros(factor_shape, dtype=tf.float32)
        zoom_heights = tf.zeros(factor_shape, dtype=tf.float32)
        zoom_widths = tf.zeros(factor_shape, dtype=tf.float32)
        shear_heights = tf.zeros(factor_shape, dtype=tf.float32)
        shear_widths = tf.zeros(factor_shape, dtype=tf.float32)

        if self._enable_rotation:
            angles = self.rotation_factor(factor_shape, dtype=tf.float32)
        if self._enable_translation:
            translation_heights = self.translation_height_factor(
                factor_shape, dtype=tf.float32
            )
            translation_widths = self.translation_width_factor(
                factor_shape, dtype=tf.float32
            )
        if self._enable_zoom:
            zoom_heights = self.zoom_height_factor(
                factor_shape, dtype=tf.float32
            )
            if self.same_zoom_factor:
                zoom_widths = zoom_heights
            else:
                zoom_widths = self.zoom_width_factor(
                    factor_shape, dtype=tf.float32
                )
        if self._enable_shear:
            shear_heights = self.shear_height_factor(
                factor_shape, dtype=tf.float32
            )
            shear_widths = self.shear_width_factor(
                factor_shape, dtype=tf.float32
            )

        angles = angles / 360.0 * 2.0 * math.pi
        translations = tf.concat(
            [translation_widths, translation_heights], axis=1
        )
        zooms = tf.concat([zoom_widths, zoom_heights], axis=1)
        shears = tf.concat([shear_widths, shear_heights], axis=1)

        # start from identity matrixes:
        #     [[1 0 0]
        #      [0 1 0]
        #      [0 0 1]]
        identity_matrixes = tf.concat(
            [
                tf.ones((batch_size, 1), dtype=tf.float32),
                tf.zeros((batch_size, 3), dtype=tf.float32),
                tf.ones((batch_size, 1), dtype=tf.float32),
                tf.zeros((batch_size, 3), dtype=tf.float32),
                tf.ones((batch_size, 1), dtype=tf.float32),
            ],
            axis=1,
        )
        combined_matrixes = tf.reshape(identity_matrixes, (batch_size, 3, 3))
        # process zoom
        if self._enable_zoom:
            zoom_matrixes = augmentation_utils.get_zoom_matrix(
                zooms, heights, widths, to_square=True
            )
            combined_matrixes = zoom_matrixes @ combined_matrixes
        # process rotations
        if self._enable_rotation:
            rotation_matrixes = augmentation_utils.get_rotation_matrix(
                angles, heights, widths, to_square=True
            )
            combined_matrixes = rotation_matrixes @ combined_matrixes
        # process shear
        if self._enable_shear:
            shear_matrixes = augmentation_utils.get_shear_matrix(
                shears, to_square=True
            )
            combined_matrixes = shear_matrixes @ combined_matrixes
        # process translations
        if self._enable_translation:
            translation_matrixes = augmentation_utils.get_translation_matrix(
                translations, heights, widths, to_square=True
            )
            combined_matrixes = translation_matrixes @ combined_matrixes
        return {
            "angles": angles,
            "translations": translations,
            "zooms": zooms,
            "shears": shears,
            "combined_matrixes": combined_matrixes,  # (batch_size, 3, 3)
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
        ori_shape = images.shape
        batch_size = tf.shape(images)[0]
        combined_matrixes = transformations["combined_matrixes"]
        combined_matrixes = tf.reshape(
            combined_matrixes, shape=(batch_size, -1)
        )
        combined_matrixes = combined_matrixes[:, :-1]

        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if images.dtype == tf.bfloat16:
            images = tf.cast(images, dtype=tf.float32)
        images = preprocessing_utils.transform(
            images,
            combined_matrixes,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        images = tf.ensure_shape(images, shape=ori_shape)
        return tf.cast(images, dtype=self.compute_dtype)

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
        # cast to float32 to avoid numerical issue
        heights, widths = augmentation_utils.get_images_shape(
            raw_images, dtype=tf.float32
        )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
            dtype=tf.float32,
        )
        boxes = bounding_boxes["boxes"]
        original_bounding_boxes = bounding_boxes.copy()
        # process rotations
        if self._enable_rotation:
            origin_x = widths / 2
            origin_y = heights / 2
            angles = -transformations["angles"]
            angles = angles[:, tf.newaxis, tf.newaxis]
            # points: (batch_size, max_num_boxes, 4, 2)
            points = tf.stack(
                [
                    tf.stack([boxes[..., 0], boxes[..., 1]], axis=2),
                    tf.stack([boxes[..., 2], boxes[..., 1]], axis=2),
                    tf.stack([boxes[..., 2], boxes[..., 3]], axis=2),
                    tf.stack([boxes[..., 0], boxes[..., 3]], axis=2),
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
        if self._enable_translation:
            translations = transformations["translations"]
            translation_widths = tf.expand_dims(
                translations[:, 0:1] * widths, axis=-1
            )
            translation_heights = tf.expand_dims(
                translations[:, 1:2] * heights, axis=-1
            )
            x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=-1)
            x1s = x1s + translation_widths
            y1s = y1s + translation_heights
            x2s = x2s + translation_widths
            y2s = y2s + translation_heights
            boxes = tf.concat([x1s, y1s, x2s, y2s], axis=-1)
        # process shear
        if self._enable_shear:
            shears = transformations["shears"]
            shear_widths = tf.expand_dims(shears[:, 0:1], axis=-1)
            shear_heights = tf.expand_dims(shears[:, 1:2], axis=-1)
            _x1s, _y1s, _x2s, _y2s = tf.split(boxes, 4, axis=-1)
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
            boxes = tf.concat([x1s, y1s, x2s, y2s], axis=-1)
        # process zoom
        if self._enable_zoom:
            zooms = transformations["zooms"]
            zoom_widths = tf.expand_dims(zooms[:, 0:1], axis=-1)
            zoom_heights = tf.expand_dims(zooms[:, 1:2], axis=-1)
            x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=-1)
            x_offsets = ((tf.expand_dims(widths, axis=-1) - 1.0) / 2.0) * (
                1.0 - zoom_widths
            )
            y_offsets = ((tf.expand_dims(heights, axis=-1) - 1.0) / 2.0) * (
                1.0 - zoom_heights
            )
            x1s = (x1s - x_offsets) / zoom_widths
            x2s = (x2s - x_offsets) / zoom_widths
            y1s = (y1s - y_offsets) / zoom_heights
            y2s = (y2s - y_offsets) / zoom_heights
            boxes = tf.concat([x1s, y1s, x2s, y2s], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box_utils.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=raw_images,
        )
        bounding_boxes = bounding_box_utils.sanitize_bounding_boxes(
            bounding_boxes,
            min_area_ratio=self.bounding_box_min_area_ratio,
            max_aspect_ratio=self.bounding_box_max_aspect_ratio,
            bounding_box_format="xyxy",
            reference_bounding_boxes=original_bounding_boxes,
            images=raw_images,
            reference_images=raw_images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
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
        self, segmentation_masks, transformations, **kwargs
    ):
        original_shape = segmentation_masks.shape
        batch_size = tf.shape(segmentation_masks)[0]
        combined_matrixes = transformations["combined_matrixes"]
        combined_matrixes = tf.reshape(
            combined_matrixes, shape=(batch_size, -1)
        )
        combined_matrixes = combined_matrixes[:, :-1]

        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if segmentation_masks.dtype == tf.bfloat16:
            segmentation_masks = tf.cast(segmentation_masks, dtype=tf.float32)
        segmentation_masks = preprocessing_utils.transform(
            segmentation_masks,
            combined_matrixes,
            fill_mode=self.fill_mode,
            fill_value=0,
            interpolation="nearest",
        )
        segmentation_masks = tf.ensure_shape(
            segmentation_masks, shape=original_shape
        )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor,
                "translation_height_factor": self.translation_height_factor,
                "translation_width_factor": self.translation_width_factor,
                "zoom_height_factor": self.zoom_height_factor,
                "zoom_width_factor": self.zoom_width_factor,
                "shear_height_factor": self.shear_height_factor,
                "shear_width_factor": self.shear_width_factor,
                "same_zoom_factor": self.same_zoom_factor,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "bounding_box_min_area_ratio": self.bounding_box_min_area_ratio,  # noqa: E501
                "bounding_box_max_aspect_ratio": self.bounding_box_max_aspect_ratio,  # noqa: E501
                "seed": self.seed,
            }
        )
        return config

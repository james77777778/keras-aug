import math

import tensorflow as tf
from keras_cv.utils import fill_utils
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils.augmentation import H_AXIS
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import W_AXIS


@keras.utils.register_keras_serializable(package="keras_cv")
class RandomGridMask(VectorizedBaseRandomLayer):
    """RandomGridMask performs the Grid Mask operation on input images.

    Args:
        size_factor (float|Sequence[float]|keras_aug.FactorSampler, optional):
            The relative size for grid masks. When represented as a single
            float, the factor will be picked between ``[0.0, 0.0 + upper]``.
            Represented as d1, d2 in the paper. Defaults to
            ``(96/224, 224/224)`` which is for ImageNet classification model.
            For COCO object detection, it is set to ``(0.01, 1.0)``
        ratio_factor (float|Sequence[float]|keras_aug.FactorSampler, optional):
            The ratio from spacings to grid masks. When represented as a single
            float, the factor will be picked between ``[0.0, 0.0 + upper]``.
            Represented as ratio in the paper. Lower values make the grid size
            smaller, and higher values make the grid mask large. ``0.5``
            indicates that grid and spacing will be of equal size. Defaults to
            ``(0.6, 0.6)`` which is for ImageNet classification model. For COCO
            object detection, it is set to ``(0.5, 0.5)``
        rotation_factor (float|Sequence[float]|keras_aug.FactorSampler, optional):
            The range of the degree that will be used to rotate the grid_mask.
            When represented as a single float, the factor will be picked
            between ``[0.0 - lower, 0.0 + upper]``. A positive value means
            rotating counter clock-wise, while a negative value means
            clock-wise. Defaults to ``(-180, 180)`` which is for ImageNet
            classification model. For COCO object detection, it is set to
            ``(0, 0)``.
        fill_mode (str, optional): The fill mode. Supported values:
            ``"constant", "gaussian_noise", "random"``. Defaults to
            ``"constant"``.
        fill_value (int|float, optional): The value to be filled inside the
            gridblock when ``fill_mode="constant"``. Defaults to ``0``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `GridMask <https://arxiv.org/abs/2001.04086>`_
        - `GridMask Official Repo <https://github.com/dvlab-research/GridMask>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        size_factor=(96.0 / 224.0, 224.0 / 224.0),
        ratio_factor=(0.6, 0.6),
        rotation_factor=(-180, 180),
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(size_factor, (int, float)):
            size_factor = (0, size_factor)
        # d1 cannot be too small, the computation of grid masks might be
        # expensive
        self.size_factor = augmentation_utils.parse_factor(
            size_factor, min_value=0.01, seed=seed
        )
        if isinstance(ratio_factor, (int, float)):
            ratio_factor = (0, ratio_factor)
        self.ratio_factor = augmentation_utils.parse_factor(
            ratio_factor, seed=seed
        )
        if isinstance(rotation_factor, (int, float)):
            rotation_factor = (-rotation_factor, rotation_factor)
        self.rotation_factor = augmentation_utils.parse_factor(
            rotation_factor, min_value=-180, max_value=180, seed=seed
        )
        self._check_parameter_values(fill_mode)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

        # decide whether to enable the rotation
        self._enable_rotation = augmentation_utils.is_factor_working(
            self.rotation_factor, not_working_value=0.0
        )

    def _check_parameter_values(self, fill_mode):
        if fill_mode not in ["constant", "gaussian_noise", "random"]:
            raise ValueError(
                '`fill_mode` should be "constant", '
                '"gaussian_noise", or "random". Got '
                f"`fill_mode`={fill_mode}"
            )

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # all operations run in tf.float32 to avoid numerical issue
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )

        # mask side length
        input_diagonal_lens = tf.sqrt(tf.square(widths) + tf.square(heights))
        mask_side_lens = tf.math.ceil(input_diagonal_lens)

        # grid unit size
        smaller_side_lens = tf.where(heights < widths, heights, widths)
        unit_ratios = self.size_factor(shape=(batch_size, 1))
        unit_sizes = unit_ratios * smaller_side_lens
        unit_sizes = tf.maximum(unit_sizes, 2)  # prevent too small units
        ratios = self.ratio_factor(shape=(batch_size, 1))
        rectangle_side_lens = ratios * unit_sizes

        # sample x and y offset for grid units randomly between 0 and unit_size
        delta_xs = self._random_generator.random_uniform(shape=(batch_size, 1))
        delta_ys = self._random_generator.random_uniform(shape=(batch_size, 1))
        delta_xs = delta_xs * unit_sizes
        delta_ys = delta_ys * unit_sizes

        # randomly rotate mask
        angles = self.rotation_factor(shape=(batch_size, 1))
        angles = angles / 360.0 * 2.0 * math.pi

        return {
            "mask_side_lens": mask_side_lens,
            "rectangle_side_lens": rectangle_side_lens,
            "unit_sizes": unit_sizes,
            "delta_xs": delta_xs,
            "delta_ys": delta_ys,
            "angles": angles,
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
        masks = tf.map_fn(
            self.compute_grid_mask_single_image,
            {IMAGES: images, **transformations},
            fn_output_signature=tf.bool,
        )
        masks = tf.expand_dims(masks, axis=-1)
        # masks (batch_size, height, width, 1)

        if self._enable_rotation:
            angles = transformations.get("angles", None)
            heights, widths = augmentation_utils.get_images_shape(
                images, dtype=tf.float32
            )
            rotation_matrixes = augmentation_utils.get_rotation_matrix(
                angles, heights, widths
            )
            masks = preprocessing_utils.transform(
                tf.cast(masks, dtype=tf.float32),
                rotation_matrixes,
                fill_mode="constant",
                fill_value=0,
                interpolation="nearest",
            )
        # center crop mask
        height = tf.shape(images)[H_AXIS]
        width = tf.shape(images)[W_AXIS]
        h_diff = tf.shape(masks)[H_AXIS] - height
        w_diff = tf.shape(masks)[W_AXIS] - width
        h_start = tf.cast(h_diff / 2, tf.int32)
        w_start = tf.cast(w_diff / 2, tf.int32)
        masks = tf.image.crop_to_bounding_box(
            masks, h_start, w_start, height, width
        )
        # convert back to boolean mask
        masks = tf.cast(masks, tf.bool)
        # compute fill
        if self.fill_mode == "constant":
            fill_value = tf.fill(tf.shape(images), self.fill_value)
            fill_value = tf.cast(fill_value, dtype=self.compute_dtype)
        elif self.fill_mode == "gaussian_noise":
            fill_value = self._random_generator.random_normal(
                shape=tf.shape(images), dtype=self.compute_dtype
            )
        else:
            fill_value = self._random_generator.random_uniform(
                shape=tf.shape(images), dtype=self.compute_dtype
            )
        return tf.where(masks, fill_value, images)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def compute_grid_mask_single_image(self, inputs):
        mask_side_len = inputs.get("mask_side_lens", None)[0]
        rectangle_side_len = inputs.get("rectangle_side_lens", None)[0]
        unit_size = inputs.get("unit_sizes", None)[0]
        delta_x = inputs.get("delta_xs", None)[0]
        delta_y = inputs.get("delta_ys", None)[0]

        # grid size (number of diagonal units in grid)
        grid_size = tf.cast(mask_side_len // unit_size + 1, dtype=tf.int32)
        grid_size_range = tf.range(1, grid_size + 1)
        # diagonal corner coordinates
        unit_size_range = (
            tf.cast(grid_size_range, dtype=unit_size.dtype) * unit_size
        )
        x1 = unit_size_range - delta_x
        x0 = x1 - rectangle_side_len
        y1 = unit_size_range - delta_y
        y0 = y1 - rectangle_side_len
        # compute grid coordinates
        x0, y0 = tf.meshgrid(x0, y0)
        x1, y1 = tf.meshgrid(x1, y1)
        # flatten mesh grid
        x0 = tf.reshape(x0, [-1])
        y0 = tf.reshape(y0, [-1])
        x1 = tf.reshape(x1, [-1])
        y1 = tf.reshape(y1, [-1])
        # convert coordinates to mask
        # corners must be tf.float32 for fill_utils.corners_to_mask
        corners = tf.cast(tf.stack([x0, y0, x1, y1], axis=-1), dtype=tf.float32)
        mask_side_len = tf.cast(mask_side_len, dtype=tf.int32)
        rectangle_masks = fill_utils.corners_to_mask(
            corners, mask_shape=(mask_side_len, mask_side_len)
        )
        grid_mask = tf.reduce_any(rectangle_masks, axis=0)
        return grid_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size_factor": self.size_factor,
                "ratio_factor": self.ratio_factor,
                "rotation_factor": self.rotation_factor,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "seed": self.seed,
            }
        )
        return config

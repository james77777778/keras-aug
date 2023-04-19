"""
References:
https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/vectorized_base_image_augmentation_layer.py
- BATCHED
- IMAGES, LABELS, TARGETS, BOUNDING_BOXES, SEGMENTATION_MASKS, KEYPOINTS
- H_AXIS, W_AXIS
https://github.com/keras-team/keras-cv/blob/master/keras_cv/utils/preprocessing.py
- parse_factor
- get_rotation_matrix
- get_translation_matrix
- get_shear_matrix
"""  # noqa: E501

import enum

import tensorflow as tf
from keras_cv import core
from tensorflow import keras

H_AXIS = -3
W_AXIS = -2

IMAGES = "images"
LABELS = "labels"
TARGETS = "targets"
BOUNDING_BOXES = "bounding_boxes"
KEYPOINTS = "keypoints"
SEGMENTATION_MASKS = "segmentation_masks"
CUSTOM_ANNOTATIONS = "custom_annotations"

BATCHED = "batched"


class PaddingPosition(enum.Enum):
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    RANDOM = "random"


def get_padding_position(position):
    if isinstance(position, PaddingPosition):
        return position
    position = position.lower()
    if position not in PaddingPosition._value2member_map_.keys():
        raise NotImplementedError(
            f"Value not recognized for `position`: {position}. Supported "
            f"values are: {PaddingPosition._value2member_map_.keys()}"
        )
    return PaddingPosition._value2member_map_[position]


def get_position_params(
    tops, bottoms, lefts, rights, position, random_generator
):
    """This function supposes arguments are at `center` padding method."""
    if position == PaddingPosition.CENTER:
        # do nothing
        bottoms = bottoms
        rights = rights
        tops = tops
        lefts = lefts
    elif position == PaddingPosition.TOP_LEFT:
        bottoms += tops
        rights += lefts
        tops = tf.zeros_like(tops)
        lefts = tf.zeros_like(lefts)
    elif position == PaddingPosition.TOP_RIGHT:
        bottoms += tops
        lefts += rights
        tops = tf.zeros_like(tops)
        rights = tf.zeros_like(rights)
    elif position == PaddingPosition.BOTTOM_LEFT:
        tops += bottoms
        rights += lefts
        bottoms = tf.zeros_like(bottoms)
        lefts = tf.zeros_like(lefts)
    elif position == PaddingPosition.BOTTOM_RIGHT:
        tops += bottoms
        lefts += rights
        bottoms = tf.zeros_like(bottoms)
        rights = tf.zeros_like(rights)
    elif position == PaddingPosition.RANDOM:
        batch_size = tf.shape(tops)[0]
        original_dtype = tops.dtype
        h_pads = tf.cast(tops + bottoms, dtype=tf.float32)
        w_pads = tf.cast(lefts + rights, dtype=tf.float32)
        tops = random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        tops = tf.cast(tf.round(tops * h_pads), dtype=original_dtype)
        bottoms = tf.cast(h_pads, dtype=tf.int32) - tops
        lefts = random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        lefts = tf.cast(tf.round(lefts * w_pads), dtype=original_dtype)
        rights = tf.cast(w_pads, dtype=tf.int32) - lefts
    else:
        raise NotImplementedError(
            f"Value not recognized for `position`: {position}. Supported "
            f"values are: {PaddingPosition._value2member_map_.keys()}"
        )

    return tops, bottoms, lefts, rights


def is_factor_working(factor, not_working_value=0.0):
    if isinstance(factor, core.ConstantFactorSampler):
        if factor.value == not_working_value:
            return False
    if isinstance(factor, core.UniformFactorSampler):
        if (
            factor.lower == not_working_value
            and factor.upper == not_working_value
        ):
            return False
    return True


def get_images_shape(images, dtype=tf.int32):
    if isinstance(images, tf.RaggedTensor):
        heights = tf.reshape(images.row_lengths(), (-1, 1))
        widths = tf.reshape(
            tf.reduce_max(images.row_lengths(axis=2), 1), (-1, 1)
        )
    else:
        batch_size = tf.shape(images)[0]
        heights = tf.repeat(tf.shape(images)[H_AXIS], repeats=[batch_size])
        heights = tf.reshape(heights, shape=(-1, 1))
        widths = tf.repeat(tf.shape(images)[W_AXIS], repeats=[batch_size])
        widths = tf.reshape(widths, shape=(-1, 1))
    return tf.cast(heights, dtype=dtype), tf.cast(widths, dtype=dtype)


def expand_dict_dims(dicts, axis):
    new_dicts = {}
    for key in dicts.keys():
        tensor = dicts[key]
        new_dicts[key] = tf.expand_dims(tensor, axis=axis)
    return new_dicts


def parse_factor(
    param,
    min_value=0.0,
    max_value=1.0,
    center_value=0.5,
    param_name="factor",
    seed=None,
):
    if isinstance(param, dict):
        # For all classes missing a `from_config` implementation.
        # (RandomHue, RandomShear, etc.)
        # To be removed with addition of `keras.__internal__` namespace support
        param = keras.utils.deserialize_keras_object(param)

    if isinstance(param, core.FactorSampler):
        return param

    if isinstance(param, float) or isinstance(param, int):
        param = (center_value - param, center_value + param)

    if param[0] > param[1]:
        raise ValueError(
            f"`{param_name}[0] > {param_name}[1]`, `{param_name}[0]` must be "
            f"<= `{param_name}[1]`. Got `{param_name}={param}`"
        )
    if (min_value is not None and param[0] < min_value) or (
        max_value is not None and param[1] > max_value
    ):
        raise ValueError(
            f"`{param_name}` should be inside of range "
            f"[{min_value}, {max_value}]. Got {param_name}={param}"
        )

    if param[0] == param[1]:
        return core.ConstantFactorSampler(param[0])

    return core.UniformFactorSampler(param[0], param[1], seed=seed)


def blend(images_1, images_2, ratios):
    """Blend the images by `ratios * images_1 + (1 - ratios) * images_2`"""
    return ratios * images_1 + (1.0 - ratios) * images_2


def get_rotation_matrix(
    angles, image_height, image_width, to_square=False, name=None
):
    """Returns projective transform(s) for the given angle(s).
    Args:
        angles: A scalar angle to rotate all images by, or
            (for batches of images) a vector with an angle to rotate each image
            in the batch. The rank must be statically known
            (the shape is not `TensorShape(None)`).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        to_square: Whether to append ones to last dimension and reshape to
            (batch_size, 3, 3), defaults to False.
        name: The name of the op.
    Returns:
        A tensor of shape (num_images, 8). Projective transforms which can be
            given to operation `image_projective_transform_v2`. If one row of
            transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the
            *output* point `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with keras.backend.name_scope(name or "rotation_matrix"):
        x_offset = (image_width - 1) - (
            tf.cos(angles) * (image_width - 1)
            - tf.sin(angles) * (image_height - 1)
        )
        x_offset /= 2.0
        y_offset = (image_height - 1) - (
            tf.sin(angles) * (image_width - 1)
            + tf.cos(angles) * (image_height - 1)
        )
        y_offset /= 2.0
        num_angles = tf.shape(angles)[0]
        matrix = tf.concat(
            [
                tf.cos(angles),
                -tf.sin(angles),
                x_offset,
                tf.sin(angles),
                tf.cos(angles),
                y_offset,
                tf.zeros((num_angles, 2), tf.float32),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_angles, 1), tf.float32)], axis=1
            )
            matrix = tf.reshape(matrix, (num_angles, 3, 3))
        return matrix


def get_translation_matrix(
    translations, image_height, image_width, to_square=False, name=None
):
    """Returns projective transform(s) for the given translation(s).

    Args:
        translations: A matrix of 2-element lists representing `[dx, dy]`
            to translate for each image (for a batch of images).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        to_square: Whether to append ones to last dimension and reshape to
            (batch_size, 3, 3), defaults to False.
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)` projective transforms which can be
        given to `transform`.
    """
    with keras.backend.name_scope(name or "translation_matrix"):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        matrix = tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, tf.newaxis] * image_width,
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, tf.newaxis] * image_height,
                tf.zeros((num_translations, 2), tf.float32),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_translations, 1), tf.float32)], axis=1
            )
            matrix = tf.reshape(matrix, (num_translations, 3, 3))
        return matrix


def get_zoom_matrix(
    zooms, image_height, image_width, to_square=False, name=None
):
    """Returns projective transform(s) for the given zoom(s).

    Args:
        zooms: A matrix of 2-element lists representing `[zx, zy]` to zoom for
            each image (for a batch of images).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        to_square: Whether to append ones to last dimension and reshape to
            (batch_size, 3, 3), defaults to False.
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)`. Projective transforms which can be
            given to operation `image_projective_transform_v2`.
            If one row of transforms is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with keras.backend.name_scope(name or "zoom_matrix"):
        num_zooms = tf.shape(zooms)[0]
        # The zoom matrix looks like:
        #     [[zx 0 0]
        #      [0 zy 0]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Zoom matrices are always float32.
        x_offset = ((image_width - 1.0) / 2.0) * (1.0 - zooms[:, 0, tf.newaxis])
        y_offset = ((image_height - 1.0) / 2.0) * (
            1.0 - zooms[:, 1, tf.newaxis]
        )
        matrix = tf.concat(
            values=[
                zooms[:, 0, tf.newaxis],
                tf.zeros((num_zooms, 1), tf.float32),
                x_offset,
                tf.zeros((num_zooms, 1), tf.float32),
                zooms[:, 1, tf.newaxis],
                y_offset,
                tf.zeros((num_zooms, 2), tf.float32),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_zooms, 1), tf.float32)], axis=1
            )
            matrix = tf.reshape(matrix, (num_zooms, 3, 3))
        return matrix


def get_shear_matrix(shears, to_square=False, name=None):
    """Returns projective transform(s) for the given shear(s).

    Args:
        shears: A matrix of 2-element lists representing `[sx, sy]` to shear for
            each image (for a batch of images).
        to_square: Whether to append ones to last dimension and reshape to
            (batch_size, 3, 3), defaults to False.
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)`. Projective transforms which can be
            given to operation `image_projective_transform_v2`.
            If one row of transforms is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with keras.backend.name_scope(name or "shear_matrix"):
        num_shears = tf.shape(shears)[0]
        # The transform matrix looks like:
        # (1, x, 0)
        # (y, 1, 0)
        # (0, 0, 1)
        # where the last entry is implicit.
        matrix = tf.concat(
            values=[
                tf.ones((num_shears, 1), tf.float32),
                shears[:, 0, tf.newaxis],
                tf.zeros((num_shears, 1), tf.float32),
                shears[:, 1, tf.newaxis],
                tf.ones((num_shears, 1), tf.float32),
                tf.zeros((num_shears, 3), tf.float32),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_shears, 1), tf.float32)], axis=1
            )
            matrix = tf.reshape(matrix, (num_shears, 3, 3))
        return matrix

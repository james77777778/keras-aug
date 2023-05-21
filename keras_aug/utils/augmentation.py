import enum
from typing import Sequence

import tensorflow as tf
from tensorflow import keras

from keras_aug.core import ConstantFactorSampler
from keras_aug.core import FactorSampler
from keras_aug.core import NormalFactorSampler
from keras_aug.core import SignedNormalFactorSampler
from keras_aug.core import UniformFactorSampler

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
    tops = tf.convert_to_tensor(tops)
    bottoms = tf.convert_to_tensor(bottoms)
    lefts = tf.convert_to_tensor(lefts)
    rights = tf.convert_to_tensor(rights)

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


def parse_factor(
    param,
    min_value=0.0,
    max_value=1.0,
    center_value=0.5,
    param_name="factor",
    seed=None,
):
    if isinstance(param, FactorSampler):
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
        return ConstantFactorSampler(param[0])

    return UniformFactorSampler(param[0], param[1], seed=seed)


def is_factor_working(factor, not_working_value=0.0):
    """Check whether ``factor`` is working or not.

    Args:
        factor (int|float|Sequence[int|float]|keras_aug.FactorSampler): The
            factor to check whether it is working or not.
        not_working_value (float, optional): The value indicating not working
            status. Defaults to ``0.0``.
    """
    if factor is None:
        return False
    if isinstance(factor, (int, float)):
        if factor == not_working_value:
            return False
    elif isinstance(factor, Sequence):
        if factor[0] == factor[1] and factor[0] == not_working_value:
            return False
    elif isinstance(factor, ConstantFactorSampler):
        if factor.value == not_working_value:
            return False
    elif isinstance(factor, UniformFactorSampler):
        if (
            factor.lower == not_working_value
            and factor.upper == not_working_value
        ):
            return False
    elif isinstance(factor, (NormalFactorSampler, SignedNormalFactorSampler)):
        if factor.stddev == 0 and factor.mean == not_working_value:
            return False
    else:
        raise ValueError(
            f"Cannot recognize factor type: {factor} with type {type(factor)}"
        )
    return True


def expand_dict_dims(dicts, axis):
    new_dicts = {}
    for key in dicts.keys():
        tensor = dicts[key]
        new_dicts[key] = tf.expand_dims(tensor, axis=axis)
    return new_dicts


def get_images_shape(images, dtype=tf.int32):
    """Get ``heights`` and ``widths`` of the input images.

    Input images can be ``tf.Tensor`` or ``tf.RaggedTensor`` with the shape of
    [B, H|None, W|None, C].

    Args:
        images (tf.Tensor|tf.RaggedTensor): The input images.
        dtype (tf.dtypes.DType, optional): The dtype of the outputs. Defaults to
            ``tf.int32``.
    """
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


def cast_to(inputs, dtype):
    if IMAGES in inputs:
        inputs[IMAGES] = tf.cast(inputs[IMAGES], dtype)
    if LABELS in inputs:
        inputs[LABELS] = tf.cast(inputs[LABELS], dtype)
    if BOUNDING_BOXES in inputs:
        inputs[BOUNDING_BOXES]["boxes"] = tf.cast(
            inputs[BOUNDING_BOXES]["boxes"], dtype
        )
        inputs[BOUNDING_BOXES]["classes"] = tf.cast(
            inputs[BOUNDING_BOXES]["classes"], dtype
        )
    if SEGMENTATION_MASKS in inputs:
        inputs[SEGMENTATION_MASKS] = tf.cast(inputs[SEGMENTATION_MASKS], dtype)
    if KEYPOINTS in inputs:
        inputs[KEYPOINTS] = tf.cast(inputs[KEYPOINTS], dtype)
    if CUSTOM_ANNOTATIONS in inputs:
        raise NotImplementedError()
    return inputs


def compute_signature(inputs, dtype):
    fn_output_signature = {}
    if IMAGES in inputs:
        if isinstance(inputs[IMAGES], tf.Tensor):
            fn_output_signature[IMAGES] = tf.TensorSpec(
                inputs[IMAGES].shape[1:], dtype
            )
        else:
            fn_output_signature[IMAGES] = tf.RaggedTensorSpec(
                shape=inputs[IMAGES].shape[1:],
                ragged_rank=1,
                dtype=dtype,
            )
    if LABELS in inputs:
        fn_output_signature[LABELS] = tf.TensorSpec(
            inputs[LABELS].shape[1:], dtype
        )
    if BOUNDING_BOXES in inputs:
        fn_output_signature[BOUNDING_BOXES] = {
            "boxes": tf.RaggedTensorSpec(
                shape=[None, 4],
                ragged_rank=1,
                dtype=dtype,
            ),
            "classes": tf.RaggedTensorSpec(
                shape=[None], ragged_rank=0, dtype=dtype
            ),
        }
    if SEGMENTATION_MASKS in inputs:
        if isinstance(inputs[SEGMENTATION_MASKS], tf.Tensor):
            fn_output_signature[SEGMENTATION_MASKS] = tf.TensorSpec(
                inputs[SEGMENTATION_MASKS].shape[1:], dtype
            )
        else:
            fn_output_signature[SEGMENTATION_MASKS] = tf.RaggedTensorSpec(
                shape=inputs[SEGMENTATION_MASKS].shape[1:],
                ragged_rank=1,
                dtype=dtype,
            )
    if KEYPOINTS in inputs:
        if isinstance(inputs[KEYPOINTS], tf.Tensor):
            fn_output_signature[KEYPOINTS] = tf.TensorSpec(
                inputs[KEYPOINTS].shape[1:], dtype
            )
        else:
            fn_output_signature[KEYPOINTS] = tf.RaggedTensorSpec(
                shape=inputs[KEYPOINTS].shape[1:],
                ragged_rank=1,
                dtype=dtype,
            )
    if CUSTOM_ANNOTATIONS in inputs:
        raise NotImplementedError()
    return fn_output_signature


def blend(images_1, images_2, factors, value_range=None):
    """Blend image1 and image2 using 'factors'. Can be batched inputs.

    Factor can be above ``0.0``.  A value of ``0.0`` means only image1 is used.
    A value of ``1.0`` means only image2 is used.  A value between ``0.0`` and
    ``1.0`` means we linearly interpolate the pixel values between the two
    images.  A value greater than ``1.0`` "extrapolates" the difference
    between the two pixel values. If ``value_range`` is set, the results will be
    clipped into ``value_range``

    Args:
        image1 (tf.Tensor): First image(s).
        image2 (tf.Tensor): Second image(s).
        factor (float|tf.Tensor): The blend factor(s).
        value_range (Sequence[int|float], optional): The value range of the
            results. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """
    results = images_1 + factors * (images_2 - images_1)
    if value_range is not None:
        results = tf.clip_by_value(results, value_range[0], value_range[1])
    return results


def rgb_to_grayscale(images):
    """Converts images from RGB to Grayscale.

    Compared to ``tf.image.rgb_to_grayscale``, this function replaces
    ``tf.tensordot`` with ``tf.math.multiply`` and ``tf.math.add`` to reduce
    memory usage.

    Args:
        images (tf.Tensor): The RGB tensor to convert. The last dimension must
            have size 3 and should contain RGB values.

    References:
        - `torchvision <https://github.com/pytorch/vision>`_
    """
    return (
        images[..., 0:1] * 0.2989
        + images[..., 1:2] * 0.587
        + images[..., 2:3] * 0.114
    )


def get_rotation_matrix(
    angles, image_height, image_width, to_square=False, name=None
):
    """Returns projective transforms for the given angles.

    Args:
        angles (tf.Tensor): a vector with the angles to rotate each image
            in the batch.
        image_height (tf.Tensor): Height of the images to be transformed.
        image_width (tf.Tensor): Width of the images to be transformed.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
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
                tf.zeros((num_angles, 2), angles.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_angles, 1), angles.dtype)], axis=1
            )
            matrix = tf.reshape(matrix, (num_angles, 3, 3))
        return matrix


def get_translation_matrix(
    translations, image_height, image_width, to_square=False, name=None
):
    """Returns projective transforms for the given translations.

    Args:
        translations (tf.Tensor): A matrix of 2-element lists representing
            ``[dx, dy]`` to translate for a batch of images.
        image_height (tf.Tensor): Height of the images to be transformed.
        image_width (tf.Tensor): Width of the images to be transformed.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
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
                tf.ones((num_translations, 1), translations.dtype),
                tf.zeros((num_translations, 1), translations.dtype),
                -translations[:, 0, tf.newaxis] * image_width,
                tf.zeros((num_translations, 1), translations.dtype),
                tf.ones((num_translations, 1), translations.dtype),
                -translations[:, 1, tf.newaxis] * image_height,
                tf.zeros((num_translations, 2), translations.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_translations, 1), translations.dtype)],
                axis=1,
            )
            matrix = tf.reshape(matrix, (num_translations, 3, 3))
        return matrix


def get_zoom_matrix(
    zooms, image_height, image_width, to_square=False, name=None
):
    """Returns projective transforms for the given zooms.

    Args:
        zooms (tf.Tensor): A matrix of 2-element lists representing
            ``[zx, zy]`` to zoom for a batch of images.
        image_height (tf.Tensor): Height of the images to be transformed.
        image_width (tf.Tensor): Width of the images to be transformed.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
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
                tf.zeros((num_zooms, 1), zooms.dtype),
                x_offset,
                tf.zeros((num_zooms, 1), zooms.dtype),
                zooms[:, 1, tf.newaxis],
                y_offset,
                tf.zeros((num_zooms, 2), zooms.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_zooms, 1), zooms.dtype)], axis=1
            )
            matrix = tf.reshape(matrix, (num_zooms, 3, 3))
        return matrix


def get_shear_matrix(shears, to_square=False, name=None):
    """Returns projective transforms for the given shears.

    Args:
        shears (tf.Tensor): A matrix of 2-element lists representing `[sx, sy]`
            to shear for a batch of images.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
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
                tf.ones((num_shears, 1), shears.dtype),
                shears[:, 0, tf.newaxis],
                tf.zeros((num_shears, 1), shears.dtype),
                shears[:, 1, tf.newaxis],
                tf.ones((num_shears, 1), shears.dtype),
                tf.zeros((num_shears, 3), shears.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_shears, 1), shears.dtype)], axis=1
            )
            matrix = tf.reshape(matrix, (num_shears, 3, 3))
        return matrix

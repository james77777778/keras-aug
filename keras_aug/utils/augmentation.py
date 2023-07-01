import enum
from typing import Sequence

import tensorflow as tf

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


def random_inversion(rng):
    negate = rng.random_uniform((), 0, 1, dtype=tf.float32) > 0.5
    negate = tf.cond(negate, lambda: -1.0, lambda: 1.0)
    return negate


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        inputs = tf.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = tf.cast(inputs, dtype)
    return inputs


_TF_INTERPOLATION_METHODS = {
    "bilinear": tf.image.ResizeMethod.BILINEAR,
    "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    "bicubic": tf.image.ResizeMethod.BICUBIC,
    "area": tf.image.ResizeMethod.AREA,
    "lanczos3": tf.image.ResizeMethod.LANCZOS3,
    "lanczos5": tf.image.ResizeMethod.LANCZOS5,
    "gaussian": tf.image.ResizeMethod.GAUSSIAN,
    "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
}


def get_interpolation(interpolation):
    interpolation = interpolation.lower()
    if interpolation not in _TF_INTERPOLATION_METHODS:
        raise NotImplementedError(
            "Value not recognized for `interpolation`: {}. Supported values "
            "are: {}".format(interpolation, _TF_INTERPOLATION_METHODS.keys())
        )
    return _TF_INTERPOLATION_METHODS[interpolation]


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            " Want fillmode  to be one of `reflect`, `wrap`, "
            "`constant` or `nearest`. Got `fill_mode` {}. ".format(fill_mode)
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            "Unknown `interpolation` {}. Only `nearest` and "
            "`bilinear` are supported.".format(interpolation)
        )

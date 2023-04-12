import enum
from typing import Sequence

import tensorflow as tf
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    BATCHED,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    BOUNDING_BOXES,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    H_AXIS,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    IMAGES,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    KEYPOINTS,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    LABELS,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    SEGMENTATION_MASKS,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    TARGETS,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    W_AXIS,
)


class PaddingPosition(enum.Enum):
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    RANDOM = "random"


PADDING_POSITION = {
    "center": PaddingPosition.CENTER,
    "top_left": PaddingPosition.TOP_LEFT,
    "top_right": PaddingPosition.TOP_RIGHT,
    "bottom_left": PaddingPosition.BOTTOM_LEFT,
    "bottom_right": PaddingPosition.BOTTOM_RIGHT,
    "random": PaddingPosition.RANDOM,
}


def get_images_shape(images):
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
    return tf.cast(heights, dtype=tf.int32), tf.cast(widths, dtype=tf.int32)


def get_padding_position(position):
    position = position.lower()
    if position not in PADDING_POSITION.keys():
        raise NotImplementedError(
            f"Value not recognized for `position`: {position}. Supported "
            f"values are: {PADDING_POSITION.keys()}"
        )
    return PADDING_POSITION[position]


def expand_dict_dims(dicts, axis):
    new_dicts = {}
    for key in dicts.keys():
        tensor = dicts[key]
        new_dicts[key] = tf.expand_dims(tensor, axis=axis)
    return new_dicts


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple.

    Args:
        param: Input value.
            If value is scalar, return value would be
                (offset - value, offset + value).
            If value is tuple, return value would be
                value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")
    if param is None:
        return param
    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        if len(param) != 2:
            raise ValueError("to_tuple expects 1 or 2 values")
        param = tuple(param)
    else:
        raise ValueError(
            "Argument param must be either scalar (int, float) or tuple"
        )
    if bias is not None:
        return tuple(bias + x for x in param)
    return tuple(param)

import numbers
from collections.abc import Sequence

from keras import backend


def standardize_parameter(
    parameter,
    name="parameter",
    center=0.0,
    bound=None,
    allow_none=True,
    allow_single_number=True,
):
    if parameter is None and not allow_none:
        raise ValueError(f"`{name}` cannot be `None`")
    if parameter is None and allow_none:
        return parameter

    if not isinstance(parameter, Sequence) and not allow_single_number:
        raise ValueError(
            f"`{name}` cannot be a single number."
            f"Received: {name}={parameter}"
        )
    if not isinstance(parameter, Sequence):
        parameter = abs(parameter)
        parameter = (center - parameter, center + parameter)
    elif len(parameter) > 2:
        raise ValueError(
            f"`{name}` must be a sequence of 2 values. "
            f"Received: {name}={parameter}"
        )
    if parameter[0] > parameter[1]:
        raise ValueError(
            f"`{name}` must be in the order that first element is bigger "
            f"that second element. Received: {name}={parameter}"
        )
    if bound is not None:
        if parameter[0] < bound[0] or parameter[1] > bound[1]:
            raise ValueError(
                f"{name} is out of bounds `[{bound[0]}, {bound[1]}]`. "
                f"Received: {name}={parameter}"
            )
    return tuple(parameter)


def standardize_value_range(value_range):
    if not isinstance(value_range, Sequence) or len(value_range) != 2:
        raise ValueError(
            "`value_range` must be a sequence of numbers. "
            f"Received: value_range={value_range}"
        )
    if value_range[0] > value_range[1]:
        raise ValueError(
            "`value_range` must be in the order that first element is bigger "
            f"that second element. Received: value_range={value_range}"
        )
    return tuple(value_range)


def standardize_size(size):
    if isinstance(size, numbers.Number):
        return int(size), int(size)
    if isinstance(size, Sequence) and len(size) == 1:
        return int(size[0]), int(size[0])
    if len(size) != 2:
        raise ValueError(
            "`size` must be a single integer or the sequence of 2 "
            f"numbers. Received: size={size}"
        )
    return int(size[0]), int(size[1])


def standardize_interpolation(interpolation):
    if isinstance(interpolation, str):
        interpolation = interpolation.lower()
        if interpolation not in ("nearest", "bilinear", "bicubic"):
            raise ValueError(
                "Invalid `interpolation`. Available values are 'nearest', "
                "'bilinear' and 'bicubic'. "
                f"Received: interpolation={interpolation}"
            )
        return interpolation
    else:
        raise ValueError(
            "`interpolation` must be `str`. "
            f"Received: interpolation={interpolation} of type "
            f"{type(interpolation)}"
        )


def standardize_padding_mode(padding_mode):
    available_padding_mode = ("constant", "reflect", "symmetric")
    if padding_mode not in available_padding_mode:
        raise ValueError(
            "Invalid `padding_mode`. Available values are: "
            f"{list(available_padding_mode)}. "
            f"Received: padding_mode={padding_mode}"
        )
    return padding_mode


def standardize_bbox_format(bounding_box_format):
    if bounding_box_format is None:
        return bounding_box_format
    available_bounding_box_format = (
        "xyxy",
        "xywh",
        "center_xywh",
        "rel_xyxy",
        "rel_xywh",
        "rel_center_xywh",
    )
    if bounding_box_format not in available_bounding_box_format:
        raise ValueError(
            "Invalid `bounding_box_format`. Available values are: "
            f"{list(available_bounding_box_format)}. "
            f"Received: bounding_box_format={bounding_box_format}"
        )
    return bounding_box_format


def standardize_data_format(data_format):
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in ("channels_last", "channels_first"):
        raise ValueError(
            "Invalid `data_format`. Available values are: "
            f"['channels_last', 'channels_first']. "
            f"Received: data_format={data_format}"
        )
    return data_format

import numbers
from collections.abc import Sequence


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

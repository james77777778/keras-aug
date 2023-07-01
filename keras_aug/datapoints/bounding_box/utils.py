"""
Contains functions to compute ious of bounding boxes.

Most of these codes come from KerasCV.
"""
import tensorflow as tf

from keras_aug.datapoints import bounding_box


def _relative_area(boxes, bounding_box_format):
    boxes = bounding_box.convert_format(
        boxes,
        source=bounding_box_format,
        target="rel_xywh",
        dtype=boxes.dtype,
    )
    widths = boxes[..., 2]
    heights = boxes[..., 3]
    # handle corner case where shear performs a full inversion.
    return tf.where(
        tf.math.logical_and(widths > 0, heights > 0), widths * heights, 0.0
    )


def _compute_area(box):
    """Computes area for bounding boxes

    Args:
      box: [N, 4] or [batch_size, N, 4] float Tensor, either batched
        or unbatched boxes.
    Returns:
      a float Tensor of [N] or [batch_size, N]
    """
    y_min, x_min, y_max, x_max = tf.split(
        box[..., :4], num_or_size_splits=4, axis=-1
    )
    return tf.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def _compute_intersection(boxes1, boxes2):
    """Computes intersection area between two sets of boxes.

    Args:
      boxes1: [N, 4] or [batch_size, N, 4] float Tensor boxes.
      boxes2: [M, 4] or [batch_size, M, 4] float Tensor boxes.
    Returns:
      a [N, M] or [batch_size, N, M] float Tensor.
    """
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        boxes1[..., :4], num_or_size_splits=4, axis=-1
    )
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        boxes2[..., :4], num_or_size_splits=4, axis=-1
    )
    boxes2_rank = len(boxes2.shape)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
    # [N, M] or [batch_size, N, M]
    intersect_ymax = tf.minimum(y_max1, tf.transpose(y_max2, perm))
    intersect_ymin = tf.maximum(y_min1, tf.transpose(y_min2, perm))
    intersect_xmax = tf.minimum(x_max1, tf.transpose(x_max2, perm))
    intersect_xmin = tf.maximum(x_min1, tf.transpose(x_min2, perm))

    intersect_height = intersect_ymax - intersect_ymin
    intersect_width = intersect_xmax - intersect_xmin
    zeros_t = tf.cast(0, intersect_height.dtype)
    intersect_height = tf.maximum(zeros_t, intersect_height)
    intersect_width = tf.maximum(zeros_t, intersect_width)

    return intersect_height * intersect_width


def _format_inputs(boxes, classes, images):
    boxes_rank = len(boxes.shape)
    if boxes_rank > 3:
        raise ValueError(
            "Expected len(boxes.shape)=2, or len(boxes.shape)=3, got "
            f"len(boxes.shape)={boxes_rank}"
        )
    boxes_includes_batch = boxes_rank == 3
    # Determine if images needs an expand_dims() call
    if images is not None:
        images_rank = len(images.shape)
        if images_rank > 4:
            raise ValueError(
                "Expected len(images.shape)=2, or len(images.shape)=3, got "
                f"len(images.shape)={images_rank}"
            )
        images_include_batch = images_rank == 4
        if boxes_includes_batch != images_include_batch:
            raise ValueError(
                "clip_to_image() expects both boxes and images to be batched, "
                "or both boxes and images to be unbatched. Received "
                f"len(boxes.shape)={boxes_rank}, "
                f"len(images.shape)={images_rank}. Expected either "
                "len(boxes.shape)=2 AND len(images.shape)=3, or "
                "len(boxes.shape)=3 AND len(images.shape)=4."
            )
        if not images_include_batch:
            images = tf.expand_dims(images, axis=0)

    if not boxes_includes_batch:
        return (
            tf.expand_dims(boxes, axis=0),
            tf.expand_dims(classes, axis=0),
            images,
            True,
        )
    return boxes, classes, images, False


def _format_outputs(boxes, classes, squeeze):
    if squeeze:
        return tf.squeeze(boxes, axis=0), tf.squeeze(classes, axis=0)
    return boxes, classes


def is_relative(bounding_box_format):
    """A util to check if a bounding box format uses relative coordinates"""
    if bounding_box_format.lower() not in bounding_box.TO_XYXY_CONVERTERS:
        raise ValueError(
            "`is_relative()` received an unsupported format for the argument "
            f"`bounding_box_format`. `bounding_box_format` should be one of "
            f"{bounding_box.TO_XYXY_CONVERTERS.keys()}. "
            f"Got bounding_box_format={bounding_box_format}"
        )

    return bounding_box_format.startswith("rel")


def as_relative(bounding_box_format):
    """A util to get the relative equivalent of a provided bounding box format.

    If the specified format is already a relative format,
    it will be returned unchanged.
    """

    if not is_relative(bounding_box_format):
        return "rel_" + bounding_box_format

    return bounding_box_format


def clip_to_image(
    bounding_boxes,
    bounding_box_format,
    images=None,
    image_shape=None,
):
    """clips bounding boxes to image boundaries.

    `clip_to_image()` clips bounding boxes that have coordinates out of bounds
    of an image down to the boundaries of the image. This is done by converting
    the bounding box to relative formats, then clipping them to the `[0, 1]`
    range. Additionally, bounding boxes that end up with a zero area have their
    class ID set to -1, indicating that there is no object present in them.

    Args:
        bounding_boxes: bounding box tensor to clip.
        bounding_box_format: the KerasCV bounding box format the bounding boxes
            are in.
        images: list of images to clip the bounding boxes to.
        image_shape: the shape of the images to clip the bounding boxes to.
    """
    boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]

    boxes = bounding_box.convert_format(
        boxes,
        source=bounding_box_format,
        target="rel_xyxy",
        images=images,
        image_shape=image_shape,
        dtype=boxes.dtype,
    )
    boxes, classes, images, squeeze = _format_inputs(boxes, classes, images)
    x1, y1, x2, y2 = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    clipped_bounding_boxes = tf.concat(
        [
            tf.clip_by_value(x1, clip_value_min=0, clip_value_max=1),
            tf.clip_by_value(y1, clip_value_min=0, clip_value_max=1),
            tf.clip_by_value(x2, clip_value_min=0, clip_value_max=1),
            tf.clip_by_value(y2, clip_value_min=0, clip_value_max=1),
        ],
        axis=-1,
    )
    areas = _relative_area(
        clipped_bounding_boxes, bounding_box_format="rel_xyxy"
    )
    clipped_bounding_boxes = bounding_box.convert_format(
        clipped_bounding_boxes,
        source="rel_xyxy",
        target=bounding_box_format,
        images=images,
        image_shape=image_shape,
        dtype=clipped_bounding_boxes.dtype,
    )
    clipped_bounding_boxes = tf.where(
        tf.expand_dims(areas > 0.0, axis=-1),
        clipped_bounding_boxes,
        tf.constant(-1, dtype=clipped_bounding_boxes.dtype),
    )
    classes = tf.where(areas > 0.0, classes, tf.constant(-1, classes.dtype))
    nan_indices = tf.math.reduce_any(
        tf.math.is_nan(clipped_bounding_boxes), axis=-1
    )
    classes = tf.where(nan_indices, tf.constant(-1, classes.dtype), classes)

    # TODO update dict and return
    clipped_bounding_boxes, classes = _format_outputs(
        clipped_bounding_boxes, classes, squeeze
    )

    result = bounding_boxes.copy()
    result["boxes"] = clipped_bounding_boxes
    result["classes"] = classes
    return result


def sanitize_bounding_boxes(
    bounding_boxes,
    min_size=None,
    min_area_ratio=None,
    max_aspect_ratio=None,
    bounding_box_format=None,
    reference_bounding_boxes=None,
    images=None,
    reference_images=None,
):
    """Sanitize bounding boxes by min_size, min_area_ratio and max_aspect_ratio.

    Args:
        bounding_boxes (dict[str, tf.Tensor]): The bounding boxes to sanitize.
        min_size (float, optional): The minimum size of the bounding boxes.
        min_area_ratio (float, optional): The minimum area ratio of original
            bounding boxes to bounding boxes for sanitizing.
            Defaults to ``None``.
        max_aspect_ratio (float, optional): The maximum aspect ratio of
            bounding boxes for sanitizing. Defaults to ``None``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/james77777778/keras-aug/blob/main/keras_aug/datapoints/bounding_box/converter.py
            for more details on supported bounding box formats.
        reference_bounding_boxes (dict[str, tf.Tensor], optional): The
            reference bounding boxes when apply sanitizing with
            min_area_ratio enabled. Defaults to ``None``.
        images (tf.Tensor, optional): The images for bounding boxes
            format conversion.
        reference_images (tf.Tensor, optional): The reference images for
            reference bounding boxes format conversion.

    References:
        - `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
    """  # noqa: E501
    if min_size is None and min_area_ratio is None and max_aspect_ratio is None:
        return bounding_boxes
    if min_size is not None:
        assert isinstance(min_size, (int, float))
    if min_area_ratio is not None:
        assert isinstance(min_area_ratio, (int, float))
    if max_aspect_ratio is not None:
        assert isinstance(max_aspect_ratio, (int, float))

    sanitize_mask = tf.zeros(
        tf.shape(bounding_boxes["boxes"])[:2], dtype=tf.bool
    )

    if min_size is not None:
        if bounding_box_format is None or images is None:
            raise ValueError(
                "When apply sanitize_bounding_boxes with "
                "min_size, must pass bounding_box_format and images."
            )
        boxes = bounding_boxes["boxes"]
        boxes = bounding_box.convert_format(
            boxes,
            source=bounding_box_format,
            target="xywh",
            images=images,
            dtype=boxes.dtype,
        )
        _, _, widths, heights = tf.split(boxes, 4, axis=-1)
        min_sides = tf.minimum(widths, heights)
        min_sides = tf.squeeze(min_sides, axis=-1)
        sanitize_mask = tf.math.logical_or(sanitize_mask, min_sides < min_size)

    if min_area_ratio is not None:
        if (
            bounding_box_format is None
            or reference_bounding_boxes is None
            or images is None
            or reference_images is None
        ):
            raise ValueError(
                "When apply sanitize_bounding_boxes with "
                "min_area_ratio, must pass bounding_box_format, "
                "reference_bounding_boxes, images and reference_images."
                "."
            )
        ref_boxes = reference_bounding_boxes["boxes"]
        ref_boxes = bounding_box.convert_format(
            ref_boxes,
            source=bounding_box_format,
            target="xyxy",
            images=reference_images,
            dtype=ref_boxes.dtype,
        )
        boxes = bounding_boxes["boxes"]
        boxes = bounding_box.convert_format(
            boxes,
            source=bounding_box_format,
            target="xyxy",
            images=images,
            dtype=boxes.dtype,
        )
        ref_areas = _compute_area(ref_boxes)
        areas = _compute_area(boxes)
        area_ratios = tf.math.divide_no_nan(areas, ref_areas)
        sanitize_mask = tf.math.logical_or(
            sanitize_mask, area_ratios < min_area_ratio
        )

    if max_aspect_ratio is not None:
        if bounding_box_format is None or images is None:
            raise ValueError(
                "When apply sanitize_bounding_boxes with "
                "max_aspect_ratio, must pass bounding_box_format and "
                "images."
            )
        boxes = bounding_boxes["boxes"]
        boxes = bounding_box.convert_format(
            boxes,
            source=bounding_box_format,
            target="xywh",
            images=images,
            dtype=boxes.dtype,
        )
        _, _, widths, heights = tf.split(boxes, 4, axis=-1)
        max_aspect_ratios = tf.squeeze(
            tf.maximum(
                tf.math.divide_no_nan(widths, heights),
                tf.math.divide_no_nan(heights, widths),
            ),
            axis=-1,
        )
        sanitize_mask = tf.math.logical_or(
            sanitize_mask, max_aspect_ratios > max_aspect_ratio
        )

    # set classes == -1
    bounding_boxes = bounding_boxes.copy()
    bounding_boxes["classes"] = tf.where(
        sanitize_mask,
        tf.constant(-1, bounding_boxes["classes"].dtype),
        bounding_boxes["classes"],
    )
    return bounding_boxes

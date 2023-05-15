import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.bounding_box.iou import _compute_area


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
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
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
        )
        boxes = bounding_boxes["boxes"]
        boxes = bounding_box.convert_format(
            boxes,
            source=bounding_box_format,
            target="xyxy",
            images=images,
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
        -1.0,
        bounding_boxes["classes"],
    )
    return bounding_boxes

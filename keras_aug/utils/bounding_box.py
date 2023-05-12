import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.bounding_box.iou import _compute_area


def sanitize_bounding_boxes(
    original_bounding_boxes,
    processed_bounding_boxes,
    area_ratio_threshold=None,
    aspect_ratio_threshold=None,
    bounding_box_format=None,
    images=None,
):
    """Sanitize bounding boxes by area_ratio_threshold and
    aspect_ratio_threshold.

    If ``area_ratio_threshold`` is provided, the bounding boxes with the
    reduced area ratio < area_ratio_threshold will be sanitized.

    If ``aspect_ratio_threshold`` is provided, the bounding boxes with the
    new aspect ratio > aspect_ratio_threshold will be sanitized.

    Args:
        original_bounding_boxes (dict[str, tf.Tensor]): The original
            bounding_boxes.
        processed_bounding_boxes (dict[str, tf.Tensor]): The bounding_boxes that
            has been processed by any operation, such as RandomAffine and
            Mosaic.
        area_ratio_threshold (float, optional): The threshold of the area ratio
            to sanitize. Defaults to ``None``.
        aspect_ratio_threshold (float, optional): The threshold of the aspect
            ratio to sanitize. Defaults to ``None``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        images (tf.Tensor, optional): The reference images for bounding boxes
            format conversion.

    Referecnes:
        - `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
    """  # noqa: E501
    if area_ratio_threshold is not None:
        assert isinstance(area_ratio_threshold, (int, float))
    if aspect_ratio_threshold is not None:
        assert isinstance(aspect_ratio_threshold, (int, float))
    assert (
        area_ratio_threshold is not None or aspect_ratio_threshold is not None
    )

    if area_ratio_threshold is not None:
        ori_boxes = original_bounding_boxes["boxes"]
        processed_boxes = processed_bounding_boxes["boxes"]
        ori_areas = _compute_area(ori_boxes)
        processed_areas = _compute_area(processed_boxes)
        area_ratios = tf.math.divide_no_nan(processed_areas, ori_areas)
        # set classes == -1 if intersection_ratios < area_threshold
        processed_bounding_boxes["classes"] = tf.where(
            area_ratios < area_ratio_threshold,
            -1.0,
            processed_bounding_boxes["classes"],
        )
    if aspect_ratio_threshold is not None:
        if bounding_box_format is None or images is None:
            raise ValueError(
                "When apply sanitize_bounding_boxes with "
                "aspect_ratio_threshold, must pass bounding_box_format and "
                "images."
            )
        eps = 1e-7
        ori_boxes = original_bounding_boxes["boxes"]
        processed_boxes = processed_bounding_boxes["boxes"]
        processed_boxes = bounding_box.convert_format(
            processed_boxes,
            source=bounding_box_format,
            target="xywh",
            images=images,
        )
        _, _, widths, heights = tf.split(processed_boxes, 4, axis=-1)
        max_aspect_ratios = tf.squeeze(
            tf.maximum(
                widths / (heights + eps),
                heights / (widths + eps),
            ),
            axis=-1,
        )
        # set classes == -1 if max_aspect_ratios > aspect_ratio_threshold
        processed_bounding_boxes["classes"] = tf.where(
            max_aspect_ratios > aspect_ratio_threshold,
            -1.0,
            processed_bounding_boxes["classes"],
        )

    return processed_bounding_boxes

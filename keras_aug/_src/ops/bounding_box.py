from keras.src.utils.backend_utils import in_tf_graph

from keras_aug._src.backend.bounding_box import BoundingBoxBackend
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.ops.bounding_box"])
def convert_format(
    boxes, source: str, target: str, height=None, width=None, dtype="float32"
):
    backend = "tensorflow" if in_tf_graph() else None
    return BoundingBoxBackend(backend).convert_format(
        boxes, source, target, height=height, width=width, dtype=dtype
    )


@keras_aug_export(parent_path=["keras_aug.ops.bounding_box"])
def clip_to_images(bounding_boxes, height=None, width=None, format="xyxy"):
    backend = "tensorflow" if in_tf_graph() else None
    return BoundingBoxBackend(backend).clip_to_images(
        bounding_boxes, height=height, width=width, format=format
    )


@keras_aug_export(parent_path=["keras_aug.ops.bounding_box"])
def affine(
    boxes,
    angle,
    translate_x,
    translate_y,
    scale,
    shear_x,
    shear_y,
    height,
    width,
    center_x=None,
    center_y=None,
    format="xyxy",
):
    if format != "xyxy":
        raise NotImplementedError
    backend = "tensorflow" if in_tf_graph() else None
    return BoundingBoxBackend(backend).affine(
        boxes,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
        center_x=center_x,
        center_y=center_y,
    )


@keras_aug_export(parent_path=["keras_aug.ops.bounding_box"])
def crop(boxes, top, left, height, width, format="xyxy"):
    if format != "xyxy":
        raise NotImplementedError
    backend = "tensorflow" if in_tf_graph() else None
    return BoundingBoxBackend(backend).crop(boxes, top, left, height, width)


@keras_aug_export(parent_path=["keras_aug.ops.bounding_box"])
def pad(boxes, top, left, format="xyxy"):
    if format != "xyxy":
        raise NotImplementedError
    backend = "tensorflow" if in_tf_graph() else None
    return BoundingBoxBackend(backend).pad(boxes, top, left)

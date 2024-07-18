import numpy as np
from keras import backend
from keras import ops

from keras_aug._src.backend.bounding_box import BoundingBoxBackend
from keras_aug._src.backend.image import ImageBackend
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.visualization"])
def draw_bounding_boxes(
    images,
    bounding_boxes,
    bounding_box_format,
    color=None,
    class_mapping=None,
    thickness=1,
    font_scale=1.0,
    data_format=None,
):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Cannot import OpenCV. You can install it by "
            "`pip install opencv-python`."
        )
    if isinstance(color, str):
        if color not in ("red", "blue", "green"):
            raise ValueError(
                "If `color` is a string. It must be one of "
                "('red', 'blue', 'green'). "
                f"Received: color={color}"
            )
        color_mapping = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
        }
        color = color_mapping[color]
    class_mapping = class_mapping or {}
    thickness = int(thickness)
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if not isinstance(bounding_boxes, dict):
        raise TypeError(
            "`bounding_boxes` should be a dict. "
            f"Received: bounding_boxes={bounding_boxes} of type "
            f"{type(bounding_boxes)}"
        )
    if "boxes" not in bounding_boxes or "classes" not in bounding_boxes:
        raise ValueError(
            "`bounding_boxes` should be a dict containing 'boxes' and "
            f"'classes' keys. Received: bounding_boxes={bounding_boxes}"
        )
    if data_format == "channels_last":
        h_axis = -3
        w_axis = -2
    else:
        h_axis = -2
        w_axis = -1
    image_backend = ImageBackend()
    bbox_backend = BoundingBoxBackend()
    height = images_shape[h_axis]
    width = images_shape[w_axis]
    bounding_boxes = bounding_boxes.copy()
    bounding_boxes = bbox_backend.convert_format(
        bounding_boxes, bounding_box_format, "xyxy", height, width
    )

    # To numpy array
    images = image_backend.transform_dtype(images, images.dtype, "uint8")
    images = ops.convert_to_numpy(images)
    boxes = ops.convert_to_numpy(bounding_boxes["boxes"])
    classes = ops.convert_to_numpy(bounding_boxes["classes"])
    if "confidences" in bounding_boxes:
        confidences = ops.convert_to_numpy(bounding_boxes["confidences"])
    else:
        confidences = None

    result = []
    batch_size = images.shape[0]
    for i in range(batch_size):
        _image = images[i]
        _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
        _box = boxes[i]
        _class = classes[i]
        for box_i in range(_box.shape[0]):
            x1, y1, x2, y2 = _box[box_i].astype("int32")
            c = _class[box_i].astype("int32")
            if c == -1:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            c = int(c)
            if color is None:
                _color = _COLORS[c % len(_COLORS)]
                _color = (_color * 255).astype("uint8").tolist()
            else:
                _color = color

            # Draw bounding box
            cv2.rectangle(_image, (x1, y1), (x2, y2), _color, thickness)

            if c in class_mapping:
                label = class_mapping[c]
                if confidences is not None:
                    conf = confidences[i][box_i]
                    label = f"{label} | {conf:.2f}"

                font_x1, font_y1 = _find_text_location(
                    x1, y1, font_scale, thickness
                )
                cv2.putText(
                    _image,
                    label,
                    (font_x1, font_y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    _color,
                    thickness,
                )
        result.append(_image)
    return np.stack(result, axis=0)


def _find_text_location(x, y, font_scale, thickness):
    font_height = int(font_scale * 12)
    target_y = y - 8
    if target_y - (2 * font_height) > 0:
        return x, y - 8

    line_offset = thickness
    static_offset = 3

    return (
        x + static_offset,
        y + (2 * font_height) + line_offset + static_offset,
    )


# From YOLOX:
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/visualize.py
_COLORS = np.array(
    [
        (0.000, 0.447, 0.741),
        (0.850, 0.325, 0.098),
        (0.929, 0.694, 0.125),
        (0.494, 0.184, 0.556),
        (0.466, 0.674, 0.188),
        (0.301, 0.745, 0.933),
        (0.635, 0.078, 0.184),
        (0.300, 0.300, 0.300),
        (0.600, 0.600, 0.600),
        (1.000, 0.000, 0.000),
        (1.000, 0.500, 0.000),
        (0.749, 0.749, 0.000),
        (0.000, 1.000, 0.000),
        (0.000, 0.000, 1.000),
        (0.667, 0.000, 1.000),
        (0.333, 0.333, 0.000),
        (0.333, 0.667, 0.000),
        (0.333, 1.000, 0.000),
        (0.667, 0.333, 0.000),
        (0.667, 0.667, 0.000),
        (0.667, 1.000, 0.000),
        (1.000, 0.333, 0.000),
        (1.000, 0.667, 0.000),
        (1.000, 1.000, 0.000),
        (0.000, 0.333, 0.500),
        (0.000, 0.667, 0.500),
        (0.000, 1.000, 0.500),
        (0.333, 0.000, 0.500),
        (0.333, 0.333, 0.500),
        (0.333, 0.667, 0.500),
        (0.333, 1.000, 0.500),
        (0.667, 0.000, 0.500),
        (0.667, 0.333, 0.500),
        (0.667, 0.667, 0.500),
        (0.667, 1.000, 0.500),
        (1.000, 0.000, 0.500),
        (1.000, 0.333, 0.500),
        (1.000, 0.667, 0.500),
        (1.000, 1.000, 0.500),
        (0.000, 0.333, 1.000),
        (0.000, 0.667, 1.000),
        (0.000, 1.000, 1.000),
        (0.333, 0.000, 1.000),
        (0.333, 0.333, 1.000),
        (0.333, 0.667, 1.000),
        (0.333, 1.000, 1.000),
        (0.667, 0.000, 1.000),
        (0.667, 0.333, 1.000),
        (0.667, 0.667, 1.000),
        (0.667, 1.000, 1.000),
        (1.000, 0.000, 1.000),
        (1.000, 0.333, 1.000),
        (1.000, 0.667, 1.000),
        (0.333, 0.000, 0.000),
        (0.500, 0.000, 0.000),
        (0.667, 0.000, 0.000),
        (0.833, 0.000, 0.000),
        (1.000, 0.000, 0.000),
        (0.000, 0.167, 0.000),
        (0.000, 0.333, 0.000),
        (0.000, 0.500, 0.000),
        (0.000, 0.667, 0.000),
        (0.000, 0.833, 0.000),
        (0.000, 1.000, 0.000),
        (0.000, 0.000, 0.167),
        (0.000, 0.000, 0.333),
        (0.000, 0.000, 0.500),
        (0.000, 0.000, 0.667),
        (0.000, 0.000, 0.833),
        (0.000, 0.000, 1.000),
        (0.000, 0.000, 0.000),
        (0.143, 0.143, 0.143),
        (0.286, 0.286, 0.286),
        (0.429, 0.429, 0.429),
        (0.571, 0.571, 0.571),
        (0.714, 0.714, 0.714),
        (0.857, 0.857, 0.857),
        (0.000, 0.447, 0.741),
        (0.314, 0.717, 0.741),
        (0.50, 0.5, 0),
    ]
).astype(np.float32)

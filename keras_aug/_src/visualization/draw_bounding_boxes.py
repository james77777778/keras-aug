import cv2
import numpy as np
from keras import backend
from keras import ops

from keras_aug._src.backend.bounding_box import BoundingBoxBackend


def draw_bounding_boxes(
    images,
    bounding_boxes,
    color,
    bounding_box_format,
    class_mapping=None,
    thickness=1,
    font_scale=1.0,
    data_format=None,
):
    class_mapping = class_mapping or {}
    thickness = int(thickness)
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if data_format == "channels_last":
        h_axis = -3
        w_axis = -2
    else:
        h_axis = -2
        w_axis = -1
    bbox_backend = BoundingBoxBackend(backend.backend())
    height = images_shape[h_axis]
    width = images_shape[w_axis]
    bounding_boxes = bbox_backend.convert_format(
        bounding_boxes, bounding_box_format, "xyxy", height, width
    )

    # To numpy array
    images = ops.convert_to_numpy(images)
    images = images.astype("uint8")
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
        _box = boxes[i]
        _class = classes[i]
        for box_i in range(_box.shape[0]):
            x1, y1, x2, y2 = _box[box_i].astype("int32")
            c = _class[box_i].astype("int32")
            if c == -1:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            c = int(c)

            # Draw bounding box
            cv2.rectangle(_image, (x1, y1), (x2, y2), color, thickness)

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
                    color,
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

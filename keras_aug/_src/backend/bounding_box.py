from keras_aug._src.backend.dynamic_backend import DynamicBackend


class BoundingBoxBackend(DynamicBackend):
    def __init__(self, name=None):
        super().__init__(name=name)

    def convert_format(
        self,
        boxes,
        source: str,
        target: str,
        height=None,
        width=None,
        dtype="float32",
    ):
        """Converts `boxes` from one format to another.

        Supported formats are:
        - `"xyxy"` representing `[left, top, right, bottom]`.
        - `"xywh"` representing `[left, top, width, height]`.
        - `"center_xywh"`. representing
            `[center of x, center of y, width, height]`.
        - `"rel_xyxy"`. representing `[left, top, right, bottom]`. All values
            in `rel_xyxy` are in the range `(0, 1)`.
        - `"rel_xywh". representing `[left, top, width, height]`. All values
            in `rel_xywh` are in the range `(0, 1)`.
        - `"rel_center_xywh"`. representing
            `[center of x, center of y, width, height]`. All values
            in `rel_center_xywh` are in the range `(0, 1)`.

        Args:
            boxes: Tensor representing bounding boxes in the format specified in
                the `source` parameter. `boxes` can optionally have extra
                dimensions stacked on the final axis to store metadata. boxes
                should be a 3D Tensor, with the shape
                `[batch_size, num_boxes, 4]`.
                Alternatively, boxes can be a dictionary with key 'boxes'
                containing a Tensor matching the aforementioned spec.
            source: One of {TODO}.
                Used to specify the original format of the `boxes` parameter.
            target: One of {TODO}.
                Used to specify the destination format of the `boxes` parameter.
            dtype: the data type to use when transforming the boxes, defaults to
                `"float32"`.
        """
        if isinstance(boxes, dict):
            boxes["boxes"] = self.convert_format(
                boxes["boxes"],
                source=source,
                target=target,
                height=height,
                width=width,
                dtype=dtype,
            )
            return boxes

        to_xyxy_converters = {
            "xyxy": self._xyxy_to_xyxy,
            "xywh": self._xywh_to_xyxy,
            "center_xywh": self._center_xywh_to_xyxy,
            "rel_xyxy": self._rel_xyxy_to_xyxy,
            "rel_xywh": self._rel_xywh_to_xyxy,
            "rel_center_xywh": self._rel_center_xywh_to_xyxy,
        }
        from_xyxy_converters = {
            "xyxy": self._xyxy_to_xyxy,
            "xywh": self._xyxy_to_xywh,
            "center_xywh": self._xyxy_to_center_xywh,
            "rel_xyxy": self._xyxy_to_rel_xyxy,
            "rel_xywh": self._xyxy_to_rel_xywh,
            "rel_center_xywh": self._xyxy_to_rel_center_xywh,
        }

        ops = self.backend
        boxes_shape = ops.shape(boxes)
        if boxes_shape[-1] != 4:
            raise ValueError(
                "`boxes` must be a tensor with the last dimension of 4. "
                f"Received: boxes.shape={boxes_shape}"
            )
        source = source.lower()
        target = target.lower()
        if source not in to_xyxy_converters.keys():
            raise ValueError(
                f"Available source: {list(to_xyxy_converters.keys())}. "
                f"Received: source={source}"
            )
        if target not in from_xyxy_converters.keys():
            raise ValueError(
                f"Available target: {list(from_xyxy_converters.keys())}. "
                f"Received: target={target}"
            )
        boxes = ops.cast(boxes, dtype)
        if source == target:
            return boxes

        if source.startswith("rel_") and target.startswith("rel_"):
            source = source.replace("rel_", "", 1)
            target = target.replace("rel_", "", 1)

        to_xyxy_converter = to_xyxy_converters[source]
        from_xyxy_converter = from_xyxy_converters[target]
        in_xyxy_boxes = to_xyxy_converter(boxes, height, width)
        return from_xyxy_converter(in_xyxy_boxes, height, width)

    def clip_to_images(
        self, bounding_boxes, height=None, width=None, format="xyxy"
    ):
        if format not in ("xyxy", "rel_xyxy"):
            raise NotImplementedError
        if format == "xyxy" and (height is None or width is None):
            raise ValueError(
                "`height` and `width` must be set if `format='xyxy'`."
            )

        ops = self.backend
        boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]

        if format == "xyxy":
            x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
            x1 = ops.numpy.clip(x1, 0, width)
            y1 = ops.numpy.clip(y1, 0, height)
            x2 = ops.numpy.clip(x2, 0, width)
            y2 = ops.numpy.clip(y2, 0, height)
            boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

            areas = self._compute_area(boxes)
            areas = ops.numpy.squeeze(areas, axis=-1)
            classes = ops.numpy.where(areas > 0, classes, -1)
        elif format == "rel_xyxy":
            x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
            x1 = ops.numpy.clip(x1, 0.0, 1.0)
            y1 = ops.numpy.clip(y1, 0.0, 1.0)
            x2 = ops.numpy.clip(x2, 0.0, 1.0)
            y2 = ops.numpy.clip(y2, 0.0, 1.0)
            boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

            areas = self._compute_area(boxes)
            areas = ops.numpy.squeeze(areas, axis=-1)
            classes = ops.numpy.where(areas > 0, classes, -1)

        result = bounding_boxes.copy()
        result["boxes"] = boxes
        result["classes"] = classes
        return result

    # Converters

    def _xyxy_to_xyxy(self, boxes, height=None, width=None):
        return boxes

    def _xywh_to_xyxy(self, boxes, height=None, width=None):
        x1, y1, w, h = self.backend.numpy.split(boxes, 4, axis=-1)
        x2 = x1 + w
        y2 = y1 + h
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _center_xywh_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        cx, cy, w, h = ops.numpy.split(boxes, 4, axis=-1)
        half_w = w / 2.0
        half_h = h / 2.0
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_xyxy_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_x1, rel_y1, rel_x2, rel_y2 = ops.numpy.split(boxes, 4, axis=-1)
        x1 = rel_x1 * width
        y1 = rel_y1 * height
        x2 = rel_x2 * width
        y2 = rel_y2 * height
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_xywh_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_x1, rel_y1, rel_w, rel_h = ops.numpy.split(boxes, 4, axis=-1)
        x1 = rel_x1 * width
        y1 = rel_y1 * height
        x2 = (rel_x1 + rel_w) * width
        y2 = (rel_y1 + rel_h) * height
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _rel_center_xywh_to_xyxy(self, boxes, height=None, width=None):
        ops = self.backend
        rel_cx, rel_cy, rel_w, rel_h = ops.numpy.split(boxes, 4, axis=-1)
        half_rel_w = rel_w / 2.0
        half_rel_h = rel_h / 2.0
        x1 = (rel_cx - half_rel_w) * height
        y1 = (rel_cy - half_rel_h) * width
        x2 = (rel_cx + half_rel_w) * height
        y2 = (rel_cy + half_rel_h) * width
        return self.backend.numpy.concatenate([x1, y1, x2, y2], axis=-1)

    def _xyxy_to_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        w = x2 - x1
        h = y2 - y1
        return self.backend.numpy.concatenate([x1, y1, w, h], axis=-1)

    def _xyxy_to_center_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        cx = x1 + ((x2 - x1) / 2.0)
        cy = y1 + ((y2 - y1) / 2.0)
        w = x2 - x1
        h = y2 - y1
        return self.backend.numpy.concatenate([cx, cy, w, h], axis=-1)

    def _xyxy_to_rel_xyxy(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_x1 = x1 / width
        rel_y1 = y1 / height
        rel_x2 = x2 / width
        rel_y2 = y2 / height
        return self.backend.numpy.concatenate(
            [rel_x1, rel_y1, rel_x2, rel_y2], axis=-1
        )

    def _xyxy_to_rel_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_x1 = x1 / width
        rel_y1 = y1 / height
        rel_w = (x2 - x1) / width
        rel_h = (y2 - y1) / height
        return self.backend.numpy.concatenate(
            [rel_x1, rel_y1, rel_w, rel_h], axis=-1
        )

    def _xyxy_to_rel_center_xywh(self, boxes, height=None, width=None):
        x1, y1, x2, y2 = self.backend.numpy.split(boxes, 4, axis=-1)
        rel_cx = (x1 + ((x2 - x1) / 2.0)) / width
        rel_cy = (y1 + ((y2 - y1) / 2.0)) / height
        rel_w = (x2 - x1) / width
        rel_h = (y2 - y1) / height
        return self.backend.numpy.concatenate(
            [rel_cx, rel_cy, rel_w, rel_h], axis=-1
        )

    # Clip

    def _compute_area(self, boxes, format="xyxy"):
        if format not in ("xyxy", "rel_xyxy"):
            raise NotImplementedError

        ops = self.backend
        x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
        widths = x2 - x1
        heights = y2 - y1
        return widths * heights

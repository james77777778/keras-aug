import typing

import keras
from keras import backend
from keras.src.utils.backend_utils import in_tf_graph

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_bbox_format
from keras_aug._src.utils.argument_validation import standardize_data_format
from keras_aug._src.utils.argument_validation import standardize_interpolation
from keras_aug._src.utils.argument_validation import standardize_padding_mode
from keras_aug._src.utils.argument_validation import standardize_size


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class Mosaic(VisionRandomLayer):
    """Apply Mosaic augmentation to the provided batch of images and labels.

    The Mosaic data augmentation takes exactly 4 images from the batch and
    makes a grid. Please ensure that the batch size is 4; any additional images
    will be ignored.

    Note that `Mosaic` is meant to be used on batches of inputs, not individual
    input. The sample pairing is deterministic and done by matching consecutive
    samples in the batch, so the batch needs to be shuffled.

    Typically, `Mosaic` expects the `labels` to be one-hot-encoded format. If
    they are not, with provided `num_classes`, this layer will transform the
    `labels` into one-hot-encoded format. (e.g. `(batch_size, num_classes)`)

    Args:
        size: Desired output size of the mosaic. If `size` is an int instead of
            sequence like `(h, w)`, a square `(size, size)` is made.
        offset: The offset of the mosaic center from the top-left corner of the
            mosaic. Defaults to `(0.25, 0.75)`.
        num_classes: The number of classes in the inputs. Used for one-hot
            encoding. Can be `None` if the labels are already one-hot-encoded.
            Defaults to `None`.
        interpolation: The interpolation mode. Available values are:
            `"nearest", "bilinear"`. Defaults to `"bilinear"`.
        padding_mode: The mode of the padding. Available values:
            `"constant", "edge", "reflect", "symmetric"`. Defaults to
            `"constant"`.
        padding_value: The padding value. It only affects when
            `padding_mode="constant"`. Defaults to `0`.
        bounding_box_format: The format of the bounding boxes. If specified,
            the available values are `"xyxy", "xywh", "center_xywh", "rel_xyxy",
            "rel_xywh", "rel_center_xywh"`. Defaults to `None`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """  # noqa: E501

    def __init__(
        self,
        size: typing.Union[typing.Sequence[int], int],
        offset: typing.Sequence[float] = (0.25, 0.75),
        num_classes: typing.Optional[int] = None,
        interpolation: str = "bilinear",
        padding_mode: str = "constant",
        padding_value: float = 0,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Check
        if not isinstance(offset, typing.Sequence) or len(offset) != 2:
            raise ValueError(
                "`offset` must be a sequence with length 2. "
                f"Received: offset={offset}"
            )
        self.size = standardize_size(size)
        self.offset = tuple(offset)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.interpolation = standardize_interpolation(interpolation)
        self.padding_mode = standardize_padding_mode(padding_mode)
        self.padding_value = padding_value
        self.bounding_box_format = standardize_bbox_format(bounding_box_format)
        self.data_format = standardize_data_format(data_format)

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_shape(
        self,
        top_left_shape,
        top_right_shape,
        bottom_left_shape,
        bottom_right_shape,
    ):
        images_shape, segmentation_masks_shape = self._get_shape_or_spec(
            top_left_shape
        )
        images_shape = list(images_shape)
        images_shape[self.h_axis] = self.size[0]
        images_shape[self.w_axis] = self.size[1]
        if segmentation_masks_shape is not None:
            segmentation_masks_shape = list(segmentation_masks_shape)
            segmentation_masks_shape[self.h_axis] = self.size[0]
            segmentation_masks_shape[self.w_axis] = self.size[1]
        return self._set_shape(
            top_left_shape, images_shape, segmentation_masks_shape
        )

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        centers_x = ops.random.uniform(
            (batch_size,), self.offset[0], self.offset[1], seed=random_generator
        )
        centers_y = ops.random.uniform(
            (batch_size,), self.offset[0], self.offset[1], seed=random_generator
        )
        return dict(centers_x=centers_x, centers_y=centers_y)

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend

        cx = transformations["centers_x"]
        cy = transformations["centers_y"]
        images_shape = ops.shape(images)
        batch_size = images_shape[0]
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        # Affine
        translate_x = (cx - 0.5) * 2.0
        translate_y = (cy - 0.5) * 2.0
        images = self.image_backend.affine(
            images,
            ops.numpy.zeros([batch_size]),
            translate_x,  # x
            translate_y,  # y
            ops.numpy.ones([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            interpolation="bilinear",
            padding_mode=self.padding_mode,
            padding_value=self.padding_value,
            data_format=self.data_format,
        )
        # Center crop
        offset_height = ops.numpy.floor_divide((height - self.size[0]), 2)
        offset_width = ops.numpy.floor_divide((width - self.size[1]), 2)
        images = self.image_backend.crop(
            images, offset_height, offset_width, self.size[0], self.size[1]
        )
        return images

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformations,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                f"{self.__class__.__name__} was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
            )
        ops = self.backend

        images_shape = ops.shape(raw_images)
        batch_size = images_shape[0]
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=height,
            width=width,
            dtype=self.bounding_box_dtype,
        )
        cx = transformations["centers_x"]
        cy = transformations["centers_y"]

        # Affine
        translate_x = (cx - 0.5) * 2.0
        translate_y = (cy - 0.5) * 2.0
        boxes = self.bbox_backend.affine(
            bounding_boxes["boxes"],
            ops.numpy.zeros([batch_size]),
            translate_x,
            translate_y,
            ops.numpy.ones([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            height,
            width,
        )
        # Center crop
        offset_height = ops.numpy.floor_divide((height - self.size[0]), 2)
        offset_width = ops.numpy.floor_divide((width - self.size[1]), 2)
        offset_height = ops.cast(offset_height, self.bounding_box_dtype)
        offset_width = ops.cast(offset_width, self.bounding_box_dtype)
        x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
        x1 = x1 - offset_width
        y1 = y1 - offset_height
        x2 = x2 - offset_width
        y2 = y2 - offset_height
        boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = self.bbox_backend.clip_to_images(
            bounding_boxes,
            height=self.size[0],
            width=self.size[1],
            format="xyxy",
        )
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=self.size[0],
            width=self.size[1],
            dtype=self.bounding_box_dtype,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        original_dtype = segmentation_masks.dtype

        cx = transformations["centers_x"]
        cy = transformations["centers_y"]
        images_shape = ops.shape(segmentation_masks)
        batch_size = images_shape[0]
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        # Affine
        translate_x = (cx - 0.5) * 2.0
        translate_y = (cy - 0.5) * 2.0
        segmentation_masks = ops.cast(
            segmentation_masks,
            backend.result_type(segmentation_masks.dtype, float),
        )
        segmentation_masks = self.image_backend.affine(
            segmentation_masks,
            ops.numpy.zeros([batch_size]),
            translate_x,  # x
            translate_y,  # y
            ops.numpy.ones([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            interpolation="nearest",
            padding_mode="constant",
            padding_value=-1,
            data_format=self.data_format,
        )
        segmentation_masks = ops.cast(segmentation_masks, original_dtype)
        # Center crop
        offset_height = ops.numpy.floor_divide((height - self.size[0]), 2)
        offset_width = ops.numpy.floor_divide((width - self.size[1]), 2)
        segmentation_masks = self.image_backend.crop(
            segmentation_masks,
            offset_height,
            offset_width,
            self.size[0],
            self.size[1],
        )
        return segmentation_masks

    def __call__(
        self, top_left, top_right, bottom_left, bottom_right, **kwargs
    ):
        if in_tf_graph():
            self._set_backend("tensorflow")
            try:
                outputs = super(VisionRandomLayer, self).__call__(
                    top_left, top_right, bottom_left, bottom_right, **kwargs
                )
            finally:
                self._reset_backend()
            return outputs
        else:
            return super(VisionRandomLayer, self).__call__(
                top_left, top_right, bottom_left, bottom_right, **kwargs
            )

    def call(self, top_left, top_right, bottom_left, bottom_right, **kwargs):
        ops = self.backend

        def format_and_cast(inputs):
            inputs, metadata = self._format_inputs(inputs)
            inputs = self._cast_inputs(inputs)
            return inputs, metadata

        top_left, metadata_tl = format_and_cast(top_left)
        top_right, metadata_tr = format_and_cast(top_right)
        bottom_left, metadata_bl = format_and_cast(bottom_left)
        bottom_right, metadata_br = format_and_cast(bottom_right)

        if not (metadata_tl == metadata_tr == metadata_bl == metadata_br):
            raise ValueError(
                "All inputs must have the same structure such as batched or as "
                "a dict. "
                f"Received: {metadata_tl}, {metadata_tr}, {metadata_bl}, "
                f"{metadata_br}"
            )

        images_shape = ops.shape(top_left[self.IMAGES])
        if len(images_shape) == 4:
            return self._format_output(
                self._batch_augment(
                    top_left, top_right, bottom_left, bottom_right
                ),
                metadata_tl,
            )
        else:
            raise ValueError(
                "Image augmentation layers are expecting inputs to be "
                "rank 3D (unbatched) or 4D (batched) tensors. "
                f"Received: images.shape={images_shape}"
            )

    def _batch_augment(
        self, top_left, top_right, bottom_left, bottom_right, **kwargs
    ):
        images, bounding_boxes, segmentation_masks = self._concate_inputs(
            top_left, top_right, bottom_left, bottom_right
        )
        raw_images = images
        labels = keypoints = custom_annotations = None
        batch_size = self.backend.shape(images)[0]

        transformations = self.get_params(
            batch_size,
            images=images,
            labels=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_masks=segmentation_masks,
            custom_annotations=custom_annotations,
        )

        images = self.augment_images(
            images,
            transformations=transformations,
            bounding_boxes=bounding_boxes,
            labels=labels,
        )
        result = {self.IMAGES: images}

        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=images,
                raw_images=raw_images,
            )
            result[self.BOUNDING_BOXES] = bounding_boxes

        if segmentation_masks is not None:
            segmentation_masks = self.augment_segmentation_masks(
                segmentation_masks,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[self.SEGMENTATION_MASKS] = segmentation_masks

        # Cannot preserve any additional inputs unmodified by this layer.
        if len(top_left.keys() - result.keys()):
            raise NotImplementedError
        return result

    def _concate_inputs(self, top_left, top_right, bottom_left, bottom_right):
        labels = top_left.get(self.LABELS, None)
        keypoints = top_left.get(self.KEYPOINTS, None)
        custom_annotations = top_left.get(self.CUSTOM_ANNOTATIONS, None)
        if (
            labels is not None
            or keypoints is not None
            or custom_annotations is not None
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} doesn't support augmenting "
                "labels, keypoints or custom annotations."
            )

        ops = self.backend

        # Concat images
        images_tl = top_left[self.IMAGES]
        images_tr = top_right[self.IMAGES]
        images_bl = bottom_left[self.IMAGES]
        images_br = bottom_right[self.IMAGES]
        images_top = ops.numpy.concatenate(
            [images_tl, images_tr], axis=self.w_axis
        )
        images_bottom = ops.numpy.concatenate(
            [images_bl, images_br], axis=self.w_axis
        )
        images = ops.numpy.concatenate(
            [images_top, images_bottom], axis=self.h_axis
        )

        # Concat bounding boxes
        bounding_boxes = None  # Default value
        bboxes_tl = top_left.get(self.BOUNDING_BOXES, None)
        if bboxes_tl is not None:
            if self.bounding_box_format is None:
                raise ValueError(
                    f"{self.__class__.__name__} was called with bounding boxes,"
                    "but no `bounding_box_format` was specified in the "
                    "constructor."
                )
            bboxes_tr = top_right[self.BOUNDING_BOXES]
            bboxes_bl = bottom_left[self.BOUNDING_BOXES]
            bboxes_br = bottom_right[self.BOUNDING_BOXES]
            images_shape = ops.shape(images_tl)
            height = images_shape[self.h_axis]
            width = images_shape[self.w_axis]

            def compute_bboxes(bounding_boxes, position="top_left"):
                bounding_boxes = self.bbox_backend.convert_format(
                    bounding_boxes,
                    source=self.bounding_box_format,
                    target="xyxy",
                    height=height,
                    width=width,
                    dtype=self.bounding_box_dtype,
                )
                boxes = bounding_boxes["boxes"]
                classes = bounding_boxes["classes"]
                if position == "top_left":
                    return boxes, classes
                elif position == "top_right":
                    x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
                    x1 = x1 + width
                    x2 = x2 + width
                    boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)
                    return boxes, classes
                elif position == "bottom_left":
                    x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
                    y1 = y1 + height
                    y2 = y2 + height
                    boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)
                    return boxes, classes
                elif position == "bottom_right":
                    x1, y1, x2, y2 = ops.numpy.split(boxes, 4, axis=-1)
                    x1 = x1 + width
                    y1 = y1 + height
                    x2 = x2 + width
                    y2 = y2 + height
                    boxes = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)
                    return boxes, classes
                else:
                    raise ValueError

            boxes_tl, classes_tl = compute_bboxes(bboxes_tl, "top_left")
            boxes_tr, classes_tr = compute_bboxes(bboxes_tr, "top_right")
            boxes_bl, classes_bl = compute_bboxes(bboxes_bl, "bottom_left")
            boxes_br, classes_br = compute_bboxes(bboxes_br, "bottom_right")
            boxes = ops.numpy.concatenate(
                [boxes_tl, boxes_tr, boxes_bl, boxes_br], axis=1
            )
            classes = ops.numpy.concatenate(
                [classes_tl, classes_tr, classes_bl, classes_br], axis=1
            )
            bounding_boxes = bboxes_tl.copy()
            bounding_boxes["boxes"] = boxes
            bounding_boxes["classes"] = classes
            bounding_boxes = self.bbox_backend.clip_to_images(
                bounding_boxes,
                height=height * 2,
                width=width * 2,
                format="xyxy",
            )
            bounding_boxes = self.bbox_backend.convert_format(
                bounding_boxes,
                source="xyxy",
                target=self.bounding_box_format,
                height=height * 2,
                width=width * 2,
                dtype=self.bounding_box_dtype,
            )

        # Concat segmentation masks
        segmentation_masks = None  # Default value
        seg_masks_tl = top_left.get(self.SEGMENTATION_MASKS, None)
        if seg_masks_tl is not None:
            seg_masks_tr = top_right[self.SEGMENTATION_MASKS]
            seg_masks_bl = bottom_left[self.SEGMENTATION_MASKS]
            seg_masks_br = bottom_right[self.SEGMENTATION_MASKS]
            seg_masks_top = ops.numpy.concatenate(
                [seg_masks_tl, seg_masks_tr], axis=self.w_axis
            )
            seg_masks_bottom = ops.numpy.concatenate(
                [seg_masks_bl, seg_masks_br], axis=self.w_axis
            )
            segmentation_masks = ops.numpy.concatenate(
                [seg_masks_top, seg_masks_bottom], axis=self.h_axis
            )

        return images, bounding_boxes, segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "offset": self.offset,
                "num_classes": self.num_classes,
                "interpolation": self.interpolation,
                "padding_mode": self.padding_mode,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config

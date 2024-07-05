import math
import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_bbox_format
from keras_aug._src.utils.argument_validation import standardize_data_format
from keras_aug._src.utils.argument_validation import standardize_interpolation
from keras_aug._src.utils.argument_validation import standardize_size
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomResizedCrop(VisionRandomLayer):
    """Crop a random portion of the inputs and resize it to a given size.

    Args:
        size: Desired output size. If `size` is a sequence like `(h, w)`,
            output size will be matched to this. If `size` is an int,  a square
            output size `(size, size)` is made.
        scale: A tuple of floats specifying the lower and upper bounds for the
            random area of the crop, before resizing. The scale is defined with
            respect to the area of the original image.
        ratio: A tuple of floats specifying the lower and upper bounds for the
            random aspect ratio of the crop, before resizing.
        interpolation: The interpolation mode. Available values are:
            `"nearest", "bilinear", "bicubic"`. Defaults to `"bilinear"`.
        antialias: Whether to apply antialiasing. It only affects bilinear and
            bicubic modes. Defaults to `True`.
        along_long_edge: Whether to resize the long edge instead of the short
            edge. It only affects when `size` is a single integer. Defaults to
            `False`.
        bounding_box_format: The format of the bounding boxes. If specified,
            the available values are `"xyxy", "xywh", "center_xywh", "rel_xyxy",
            "rel_xywh", "rel_center_xywh"`. Defaults to `None`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        size: typing.Union[typing.Sequence[int], int],
        scale: typing.Tuple[float, float] = (0.08, 1.0),
        ratio: typing.Tuple[float, float] = (3 / 4, 4 / 3),
        interpolation: str = "bilinear",
        antialias: bool = True,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = standardize_size(size)
        self.scale = standardize_value_range(scale)
        self.ratio = standardize_value_range(ratio)
        self.interpolation = standardize_interpolation(interpolation)
        self.antialias = antialias
        self.bounding_box_format = standardize_bbox_format(bounding_box_format)
        self.data_format = standardize_data_format(data_format)

        self.log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_shape(self, input_shape):
        images_shape, segmentation_masks_shape = self._get_shape_or_spec(
            input_shape
        )
        images_shape = list(images_shape)
        images_shape[self.h_axis] = self.size[0]
        images_shape[self.w_axis] = self.size[1]
        if segmentation_masks_shape is not None:
            segmentation_masks_shape = list(segmentation_masks_shape)
            segmentation_masks_shape[self.h_axis] = self.size[0]
            segmentation_masks_shape[self.w_axis] = self.size[1]
        return self._set_shape(
            input_shape, images_shape, segmentation_masks_shape
        )

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        images_shape = ops.shape(images)
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        area = ops.cast(height * width, "float32")
        scale = ops.random.uniform(
            shape=(),
            minval=self.scale[0],
            maxval=self.scale[1],
            seed=random_generator,
        )
        target_area = area * scale
        log_ratio = ops.random.uniform(
            shape=(),
            minval=self.log_ratio[0],
            maxval=self.log_ratio[1],
            seed=random_generator,
        )
        aspect_ratio = ops.numpy.exp(log_ratio)

        w = ops.cast(
            ops.numpy.round(ops.numpy.sqrt(target_area * aspect_ratio)), "int32"
        )
        h = ops.cast(
            ops.numpy.round(ops.numpy.sqrt(target_area / aspect_ratio)), "int32"
        )
        w = ops.numpy.clip(w, 1, width)
        h = ops.numpy.clip(h, 1, height)
        i = ops.random.randint(
            shape=(), minval=0, maxval=(height - h + 1), seed=random_generator
        )
        j = ops.random.randint(
            shape=(), minval=0, maxval=(width - w + 1), seed=random_generator
        )
        return dict(top=i, left=j, height=h, width=w)

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        original_dtype = backend.standardize_dtype(images.dtype)
        top = transformations["top"]
        left = transformations["left"]
        height = transformations["height"]
        width = transformations["width"]

        # Crop
        images = self.image_backend.crop(
            images, top, left, height, width, self.data_format
        )

        # Resize
        images = ops.image.resize(
            images,
            (self.size[0], self.size[1]),
            interpolation=self.interpolation,
            antialias=self.antialias,
            data_format=self.data_format,
        )
        return ops.cast(images, original_dtype)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

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

        ori_images_shape = ops.shape(raw_images)
        ori_height = ori_images_shape[self.h_axis]
        ori_width = ori_images_shape[self.w_axis]
        top = ops.cast(transformations["top"], "float32")
        left = ops.cast(transformations["left"], "float32")
        h_scale = self.size[0] / ops.cast(transformations["height"], "float32")
        w_scale = self.size[1] / ops.cast(transformations["width"], "float32")
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=ori_height,
            width=ori_width,
            dtype=self.bounding_box_dtype,
        )

        x1, y1, x2, y2 = ops.numpy.split(bounding_boxes["boxes"], 4, axis=-1)
        x1 = (x1 - left) * w_scale
        y1 = (y1 - top) * h_scale
        x2 = (x2 - left) * w_scale
        y2 = (y2 - top) * h_scale
        outputs = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
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
        original_dtype = backend.standardize_dtype(segmentation_masks.dtype)
        top = transformations["top"]
        left = transformations["left"]
        height = transformations["height"]
        width = transformations["width"]

        # Crop
        segmentation_masks = self.image_backend.crop(
            segmentation_masks, top, left, height, width, self.data_format
        )

        # Resize
        segmentation_masks = ops.image.resize(
            segmentation_masks,
            (self.size[0], self.size[1]),
            interpolation="nearest",
            antialias=False,
            data_format=self.data_format,
        )
        return ops.cast(segmentation_masks, original_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "scale": self.scale,
                "ratio": self.ratio,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config

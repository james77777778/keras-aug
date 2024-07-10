import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_bbox_format
from keras_aug._src.utils.argument_validation import standardize_data_format
from keras_aug._src.utils.argument_validation import standardize_interpolation
from keras_aug._src.utils.argument_validation import standardize_padding_mode
from keras_aug._src.utils.argument_validation import standardize_parameter


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomRotation(VisionRandomLayer):
    """Randomly rotate the inputs by angle.

    Note that `degree` is in angle units.

    Args:
        degree: Range of degrees to select from. If `degree` is a number instead
            of sequence like `(min, max)`, the range of degrees will be
            `(-degree, +degree)`. Set to `None` to deactivate rotations.
            Defaults to `None`.
        center: Optional center of rotation in the format of `(rel_x, rel_y)`.
            Default is the center of the images.
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
    """

    def __init__(
        self,
        degree: typing.Union[None, float, typing.Sequence[float]] = None,
        center: typing.Optional[typing.Sequence[float]] = None,
        interpolation: str = "bilinear",
        padding_mode: str = "constant",
        padding_value: float = 0,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        center, padding_mode = self._check_arguments(center, padding_mode)
        self.degree = standardize_parameter(degree)
        self.center = center
        self.interpolation = standardize_interpolation(interpolation)
        self.padding_mode = standardize_padding_mode(padding_mode)
        self.padding_value = padding_value
        self.bounding_box_format = standardize_bbox_format(bounding_box_format)
        self.data_format = standardize_data_format(data_format)

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def _check_arguments(self, center, padding_mode):
        if center is not None:
            center = tuple(float(c) for c in center)
            if len(center) != 2:
                raise ValueError(
                    "If `center` is specified, it must be a sequence of 2 "
                    f"values. Received: center={center}"
                )

        available_padding_mode = ("constant", "reflect", "symmetric")
        if padding_mode not in available_padding_mode:
            raise ValueError(
                "Invalid `padding_mode`. Available values are: "
                f"{list(available_padding_mode)}. "
                f"Received: padding_mode={padding_mode}"
            )
        return center, padding_mode

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        if self.degree is not None:
            angle = ops.random.uniform(
                [batch_size],
                self.degree[0],
                self.degree[1],
                seed=random_generator,
            )
        else:
            angle = ops.numpy.zeros([batch_size])
        return dict(angle=angle)

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        original_dtype = images.dtype
        images_shape = ops.shape(images)
        batch_size = images_shape[0]
        height = images_shape[self.h_axis]
        width = images_shape[self.w_axis]
        if self.center is None:
            center_x, center_y = 0.5, 0.5
        else:
            center_x, center_y = self.center
        matrix = self.image_backend.compute_affine_matrix(
            center_x,
            center_y,
            transformations["angle"],
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.ones([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            height,
            width,
        )

        # Affine
        transform = ops.numpy.reshape(matrix, [-1, 9])[:, :8]
        images = self.image_backend.transform_dtype(
            images, backend.result_type(images.dtype, float)
        )
        images = ops.image.affine_transform(
            images,
            transform,
            self.interpolation,
            self.padding_mode,
            self.padding_value,
            self.data_format,
        )
        images = self.image_backend.transform_dtype(images, original_dtype)
        return ops.cast(images, self.compute_dtype)

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

        images_shape = ops.shape(raw_images)
        batch_size = images_shape[0]
        height = images_shape[self.h_axis]
        width = images_shape[self.w_axis]
        n_boxes = ops.shape(bounding_boxes["boxes"])[1]
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=height,
            width=width,
            dtype=self.bounding_box_dtype,
        )
        if self.center is None:
            center_x, center_y = 0.5, 0.5
        else:
            center_x, center_y = self.center
        matrix = self.image_backend.compute_inverse_affine_matrix(
            center_x,
            center_y,
            transformations["angle"],
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.ones([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            height,
            width,
        )
        transposed_matrix = ops.numpy.transpose(matrix[:, :2, :], [0, 2, 1])
        points = bounding_boxes["boxes"]  # [B, N, 4]
        points = ops.numpy.stack(
            [
                points[..., 0],
                points[..., 1],
                points[..., 2],
                points[..., 1],
                points[..., 2],
                points[..., 3],
                points[..., 0],
                points[..., 3],
            ],
            axis=-1,
        )
        points = ops.numpy.reshape(points, [batch_size, n_boxes, 4, 2])
        points = ops.numpy.concatenate(
            [
                points,
                ops.numpy.ones([batch_size, n_boxes, 4, 1], points.dtype),
            ],
            axis=-1,
        )
        transformed_points = ops.numpy.einsum(
            "bnxy,byz->bnxz", points, transposed_matrix
        )
        boxes_min = ops.numpy.amin(transformed_points, axis=2)
        boxes_max = ops.numpy.amax(transformed_points, axis=2)
        outputs = ops.numpy.concatenate([boxes_min, boxes_max], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
        bounding_boxes = self.bbox_backend.clip_to_images(
            bounding_boxes,
            height=height,
            width=width,
            format="xyxy",
        )
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=height,
            width=width,
            dtype=self.bounding_box_dtype,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        original_dtype = segmentation_masks.dtype
        segmentation_masks_shape = ops.shape(segmentation_masks)
        batch_size = segmentation_masks_shape[0]
        height = segmentation_masks_shape[self.h_axis]
        width = segmentation_masks_shape[self.w_axis]
        if self.center is None:
            center_x, center_y = 0.5, 0.5
        else:
            center_x, center_y = self.center
        matrix = self.image_backend.compute_affine_matrix(
            center_x,
            center_y,
            transformations["angle"],
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.ones([batch_size]),
            ops.numpy.zeros([batch_size]),
            ops.numpy.zeros([batch_size]),
            height,
            width,
        )

        # Affine
        transform = ops.numpy.reshape(matrix, [-1, 9])[:, :8]
        segmentation_masks = ops.cast(
            segmentation_masks,
            backend.result_type(segmentation_masks.dtype, float),
        )
        segmentation_masks = ops.image.affine_transform(
            segmentation_masks,
            transform,
            "nearest",
            "constant",
            -1,
            self.data_format,
        )
        segmentation_masks = ops.cast(segmentation_masks, original_dtype)
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "degree": self.degree,
                "center": self.center,
                "interpolation": self.interpolation,
                "padding_mode": self.padding_mode,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config

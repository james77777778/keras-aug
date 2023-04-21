import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils.augmentation import BATCHED
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import LABELS


@keras.utils.register_keras_serializable(package="keras_aug")
class MosaicYOLOV8(VectorizedBaseRandomLayer):
    """MosaicYOLOV8 implements the mosaic data augmentation technique used by
    YOLOV8.

    Mosaic data augmentation first takes 4 images from the batch and makes a
    grid. After that based on the offset, a crop is taken to form the mosaic
    image. Labels are in the same ratio as the area of their images in the
    output image. Bounding boxes are translated according to the position of the
    4 images.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        offset: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `offset` is used to determine the offset
            of the mosaic center from the top-left corner of the mosaic. If a
            tuple is used, the x and y coordinates of the mosaic center are
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`. Defaults to
            (0.25, 0.75).
        bounding_box_format: a case-insensitive string (for example, "xyxy") to
            be passed if bounding boxes are being augmented by this layer. Each
            bounding box is defined by at least these 4 values. The inputs may
            contain additional information such as classes and confidence after
            these 4 values but these values will be ignored and returned as is.
            For detailed information on the supported formats, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
            Defaults to None.
        seed: Used to create a random seed, defaults to None.

    References:
        - [Yolov4 paper](https://arxiv.org/pdf/2004.10934).
        - [Yolov5 implementation](https://github.com/ultralytics/yolov5).
        - [YoloX implementation](https://github.com/Megvii-BaseDetection/YOLOX)
        - [Yolov8 implementation](https://github.com/ultralytics/ultralytics)
    """  # noqa: E501

    def __init__(
        self,
        height,
        width,
        offset=(0.25, 0.75),
        padding_value=114,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        # set force_no_unwrap_ragged_image_call=True because MosaicYOLOV8 needs
        # to process images in batch.
        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        super().__init__(
            force_no_unwrap_ragged_image_call=True,
            force_output_dense_images=True,
            seed=seed,
            **kwargs,
        )
        single_image_max_size = max((height, width)) // 2
        offset = sorted(offset)

        self.height = height
        self.width = width
        self.offset = offset
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.center_sampler = preprocessing_utils.parse_factor(
            offset, param_name="offset", seed=seed
        )
        self.single_image_max_size = single_image_max_size  # for padding

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # pick 3 indices for every batch to create the mosaic output with.
        permutation_order = self._random_generator.random_uniform(
            (batch_size, 3),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32,
        )
        # concatenate the batches with permutation order to get all 4 images of
        # the mosaic
        permutation_order = tf.concat(
            [tf.expand_dims(tf.range(batch_size), axis=-1), permutation_order],
            axis=-1,
        )

        mosaic_centers_x = self.center_sampler(
            shape=(batch_size,), dtype=self.compute_dtype
        )
        mosaic_centers_y = self.center_sampler(
            shape=(batch_size,), dtype=self.compute_dtype
        )
        mosaic_centers = tf.stack((mosaic_centers_x, mosaic_centers_y), axis=-1)

        return {
            "permutation_order": permutation_order,
            "mosaic_centers": mosaic_centers,
        }

    def augment_images(self, images, transformations, **kwargs):
        is_ragged_input = isinstance(images, tf.RaggedTensor)
        permutation_order = transformations["permutation_order"]
        mosaic_images = tf.gather(images, permutation_order)
        inputs_for_pad_and_mosaic_single_image = {
            "transformations": transformations,
            IMAGES: mosaic_images,
        }
        images = tf.map_fn(
            self.get_mosaic_single_image,
            inputs_for_pad_and_mosaic_single_image,
            fn_output_signature=self.compute_dtype,
        )
        if is_ragged_input:
            images = tf.RaggedTensor.from_tensor(images)
        return images

    def augment_labels(self, labels, transformations, images=None, **kwargs):
        is_ragged_labels = isinstance(labels, tf.RaggedTensor)
        if is_ragged_labels:
            labels = labels.to_tensor()
        labels = tf.cast(labels, dtype=self.compute_dtype)
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=self.compute_dtype
        )
        # updates labels for one output mosaic
        permutation_order = transformations["permutation_order"]
        labels_for_mosaic = tf.gather(labels, permutation_order)

        mosaic_centers = transformations["mosaic_centers"]
        center_xs = tf.expand_dims(mosaic_centers[..., 0], axis=-1) * widths
        center_ys = tf.expand_dims(mosaic_centers[..., 1], axis=-1) * heights

        areas = heights * widths

        # labels are in the same ratio as the area of the images
        top_left_ratio = (center_xs * center_ys) / areas
        top_right_ratio = ((widths - center_xs) * center_ys) / areas
        bottom_left_ratio = (center_xs * (heights - center_ys)) / areas
        bottom_right_ratio = (
            (widths - center_xs) * (heights - center_ys)
        ) / areas
        labels = (
            labels_for_mosaic[:, 0] * top_left_ratio
            + labels_for_mosaic[:, 1] * top_right_ratio
            + labels_for_mosaic[:, 2] * bottom_left_ratio
            + labels_for_mosaic[:, 3] * bottom_right_ratio
        )
        if is_ragged_labels:
            labels = tf.RaggedTensor.from_tensor(labels)
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        batch_size = tf.shape(raw_images)[0]
        heights, widths = augmentation_utils.get_images_shape(
            raw_images, dtype=self.compute_dtype
        )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
            dtype=self.compute_dtype,
        )
        boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]

        # values to translate the boxes by in the mosaic image
        mosaic_centers = transformations["mosaic_centers"]
        mosaic_centers_x = tf.expand_dims(
            mosaic_centers[..., 0] * self.width, axis=-1
        )
        mosaic_centers_y = tf.expand_dims(
            mosaic_centers[..., 1] * self.height, axis=-1
        )

        # updates bounding_boxes for one output mosaic
        permutation_order = transformations["permutation_order"]
        heights_for_mosaic = tf.gather(heights, permutation_order)
        widths_for_mosaic = tf.gather(widths, permutation_order)
        classes_for_mosaic = tf.gather(classes, permutation_order)
        boxes_for_mosaic = tf.gather(boxes, permutation_order)

        # translate_xs/translate_ys 3D:
        # (batch_size, mosaic_index, 1)
        translate_xs = tf.stack(
            (
                mosaic_centers_x - widths_for_mosaic[:, 0, ...],
                mosaic_centers_x,
                mosaic_centers_x - widths_for_mosaic[:, 2, ...],
                mosaic_centers_x,
            ),
            axis=1,
        )
        translate_ys = tf.stack(
            (
                mosaic_centers_y - heights_for_mosaic[:, 0, ...],
                mosaic_centers_y - heights_for_mosaic[:, 1, ...],
                mosaic_centers_y,
                mosaic_centers_y,
            ),
            axis=1,
        )
        # translate_xs/translate_ys 4D:
        # (batch_size, mosaic_index, 1, 4)
        # 4 means translates for xyxy
        translates = tf.stack(
            (translate_xs, translate_ys, translate_xs, translate_ys), axis=-1
        )
        # boxes_for_mosaic 4D:
        # (batch_size, mosaic_index, num_boxes, coordinates)
        boxes_for_mosaic += translates

        boxes_for_mosaic = tf.reshape(boxes_for_mosaic, [batch_size, -1, 4])
        classes_for_mosaic = tf.reshape(classes_for_mosaic, [batch_size, -1])
        boxes_for_mosaic = {
            "boxes": boxes_for_mosaic,
            "classes": classes_for_mosaic,
        }
        boxes_for_mosaic = bounding_box.clip_to_image(
            boxes_for_mosaic,
            bounding_box_format="xyxy",
            image_shape=(self.height, self.width, None),
        )
        boxes_for_mosaic = bounding_box.convert_format(
            boxes_for_mosaic,
            source="xyxy",
            target=self.bounding_box_format,
            image_shape=(self.height, self.width, None),
            dtype=self.compute_dtype,
        )
        return boxes_for_mosaic

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        return super()._batch_augment(inputs)

    def call(self, inputs):
        _, metadata = self._format_inputs(inputs)
        if metadata[BATCHED] is not True:
            raise ValueError(
                "MosaicYOLOV8 received a single image to `call`. The "
                "layer relies on combining multiple examples, and as such "
                "will not behave as expected. Please call the layer with 4 "
                "or more samples."
            )
        return super().call(inputs=inputs)

    def _validate_inputs(self, inputs):
        images = inputs.get(IMAGES, None)
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        if images is None or (labels is None and bounding_boxes is None):
            raise ValueError(
                "MosaicYOLOV8 expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}. or'
                '{"images": images, "bounding_boxes": bounding_boxes}'
                f"Got: inputs = {inputs}"
            )
        if bounding_boxes is not None and self.bounding_box_format is None:
            raise ValueError(
                "MosaicYOLOV8 received bounding boxes but no "
                "bounding_box_format. Please pass a bounding_box_format from "
                "the supported list."
            )

    def pad_and_crop_image_patch_single(
        self, image, height, width, target_height, target_width, mosaic_index
    ):
        if mosaic_index not in (0, 1, 2, 3):
            ValueError(
                "mosaic_index must be in the range of `[0, 3]`. "
                f"Received mosaic_index: {mosaic_index}"
            )
        padding_row = target_width - width
        padding_col = target_height - height
        # pad
        if mosaic_index == 0:
            paddings = [[padding_col, 0], [padding_row, 0], [0, 0]]
        elif mosaic_index == 1:
            paddings = [[padding_col, 0], [0, padding_row], [0, 0]]
        elif mosaic_index == 2:
            paddings = [[0, padding_col], [padding_row, 0], [0, 0]]
        elif mosaic_index == 3:
            paddings = [[0, padding_col], [0, padding_row], [0, 0]]
        paddings = tf.convert_to_tensor(paddings, dtype=tf.int32)
        paddings = tf.where(paddings > 0, paddings, tf.zeros_like(paddings))
        image = tf.pad(
            image, paddings=paddings, constant_values=self.padding_value
        )
        # crop
        offsets = tf.convert_to_tensor(
            [padding_col, padding_row], dtype=tf.int32
        )
        offsets = tf.where(
            offsets < 0, tf.negative(offsets), tf.zeros_like(offsets)
        )
        if mosaic_index == 0:
            offset_height = offsets[0]
            offset_width = offsets[1]
        elif mosaic_index == 1:
            offset_height = offsets[0]
            offset_width = 0
        elif mosaic_index == 2:
            offset_height = 0
            offset_width = offsets[1]
        elif mosaic_index == 3:
            offset_height = 0
            offset_width = 0
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=target_height,
            target_width=target_width,
        )
        return image

    def get_mosaic_single_image(self, inputs):
        mosaic_images = inputs.get(IMAGES, None)
        transformation = inputs.get("transformations")
        mosaic_centers = transformation["mosaic_centers"]
        # must be tf.int32 for following tf.image.crop_to_bounding_box
        mosaic_centers_x = tf.cast(
            tf.round(mosaic_centers[0] * self.width), dtype=tf.int32
        )
        mosaic_centers_y = tf.cast(
            tf.round(mosaic_centers[1] * self.height), dtype=tf.int32
        )
        heights, widths = augmentation_utils.get_images_shape(
            mosaic_images, dtype=tf.int32
        )

        # get images
        top_left_image = mosaic_images[0]
        top_right_image = mosaic_images[1]
        bottom_left_image = mosaic_images[2]
        bottom_right_image = mosaic_images[3]
        if isinstance(mosaic_images, tf.RaggedTensor):
            top_left_image = top_left_image.to_tensor()
            top_right_image = top_right_image.to_tensor()
            bottom_left_image = bottom_left_image.to_tensor()
            bottom_right_image = bottom_right_image.to_tensor()

        # top_left
        top_left_image = self.pad_and_crop_image_patch_single(
            top_left_image,
            heights[0][0],
            widths[0][0],
            mosaic_centers_y,
            mosaic_centers_x,
            mosaic_index=0,
        )
        # top_right
        top_right_image = self.pad_and_crop_image_patch_single(
            top_right_image,
            heights[1][0],
            widths[1][0],
            mosaic_centers_y,
            self.width - mosaic_centers_x,
            mosaic_index=1,
        )
        # bottom_left
        bottom_left_image = self.pad_and_crop_image_patch_single(
            bottom_left_image,
            heights[2][0],
            widths[2][0],
            self.height - mosaic_centers_y,
            mosaic_centers_x,
            mosaic_index=2,
        )
        # bottom_right
        bottom_right_image = self.pad_and_crop_image_patch_single(
            bottom_right_image,
            heights[3][0],
            widths[3][0],
            self.height - mosaic_centers_y,
            self.width - mosaic_centers_x,
            mosaic_index=3,
        )

        # concat patches
        tops = tf.concat([top_left_image, top_right_image], axis=1)
        bottoms = tf.concat([bottom_left_image, bottom_right_image], axis=1)
        outputs = tf.concat([tops, bottoms], axis=0)
        return tf.cast(outputs, dtype=self.compute_dtype)

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "offset": self.offset,
            "padding_value": self.padding_value,
            "bounding_box_format": self.bounding_box_format,
            "seed": self.seed,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
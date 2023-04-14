import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentations import ResizeLongest
from keras_aug.utils import augmentation_utils
from keras_aug.utils.augmentation_utils import BATCHED
from keras_aug.utils.augmentation_utils import BOUNDING_BOXES
from keras_aug.utils.augmentation_utils import IMAGES
from keras_aug.utils.augmentation_utils import KEYPOINTS
from keras_aug.utils.augmentation_utils import LABELS
from keras_aug.utils.augmentation_utils import SEGMENTATION_MASKS


@keras.utils.register_keras_serializable(package="keras_aug")
class MosaicYOLOV8(VectorizedBaseImageAugmentationLayer):
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
        seed: integer, used to create a random seed.
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
        super().__init__(seed=seed, **kwargs)
        single_image_max_size = max((height, width)) // 2
        offset = sorted(offset)

        self.height = height
        self.width = width
        self.offset = offset
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.resize_longest = ResizeLongest(
            max_size=single_image_max_size,
            bounding_box_format=bounding_box_format,
            seed=seed,
        )
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
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
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
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        batch_size = tf.shape(raw_images)[0]
        heights, widths = augmentation_utils.get_images_shape(
            raw_images, dtype=tf.float32
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
        images = inputs.get(IMAGES, None)
        raw_images = images
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        keypoints = inputs.get(KEYPOINTS, None)
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        batch_size = tf.shape(images)[0]
        transformations = self.get_random_transformation_batch(
            batch_size,
            images=images,
            labels=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_masks=segmentation_masks,
        )

        images = self.augment_images(
            images,
            transformations=transformations,
            bounding_boxes=bounding_boxes,
            labels=labels,
        )
        result = {IMAGES: images}

        if labels is not None:
            labels = self.augment_targets(
                labels,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[LABELS] = labels

        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=images,
                raw_images=raw_images,
            )
            bounding_boxes = bounding_box.to_ragged(bounding_boxes)
            result[BOUNDING_BOXES] = bounding_boxes

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

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
        if labels is not None and not labels.dtype.is_floating:
            raise ValueError(
                f"MosaicYOLOV8 received labels with type {labels.dtype}. "
                "Labels must be of type float."
            )
        if bounding_boxes is not None and self.bounding_box_format is None:
            raise ValueError(
                "MosaicYOLOV8 received bounding boxes but no "
                "bounding_box_format. Please pass a bounding_box_format from "
                "the supported list."
            )

    def _pad_and_crop_single_patch(
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
        mosaic_centers_x = tf.cast(
            mosaic_centers[0] * self.width, dtype=tf.int32
        )
        mosaic_centers_y = tf.cast(
            mosaic_centers[1] * self.height, dtype=tf.int32
        )
        heights, widths = augmentation_utils.get_images_shape(mosaic_images)

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
        top_left_image = self._pad_and_crop_single_patch(
            top_left_image,
            heights[0][0],
            widths[0][0],
            mosaic_centers_y,
            mosaic_centers_x,
            mosaic_index=0,
        )
        # top_right
        top_right_image = self._pad_and_crop_single_patch(
            top_right_image,
            heights[1][0],
            widths[1][0],
            mosaic_centers_y,
            self.width - mosaic_centers_x,
            mosaic_index=1,
        )
        # bottom_left
        bottom_left_image = self._pad_and_crop_single_patch(
            bottom_left_image,
            heights[2][0],
            widths[2][0],
            self.height - mosaic_centers_y,
            mosaic_centers_x,
            mosaic_index=2,
        )
        # bottom_right
        bottom_right_image = self._pad_and_crop_single_patch(
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

from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import bounding_box as bounding_box_utils
from keras_aug.utils.augmentation import BOUNDING_BOXES


@keras.utils.register_keras_serializable(package="keras_aug")
class SanitizeBoundingBox(VectorizedBaseRandomLayer):
    """Remove degenerate/invalid bounding boxes.

    Args:
        min_size (int): The minimum size of the smaller side of bounding boxes.
        bounding_box_format (str): The format of bounding boxes of input
            dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.

    References:
        - `torchvision <https://github.com/pytorch/vision>`_
    """

    def __init__(self, min_size, bounding_box_format=None, **kwargs):
        super().__init__(**kwargs)
        self.min_size = min_size
        self.bounding_box_format = bounding_box_format

    def augment_ragged_image(self, image, transformation, **kwargs):
        return image

    def augment_images(self, images, transformations, **kwargs):
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`SanitizeBoundingBox()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`SanitizeBoundingBox(..., bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box_utils.sanitize_bounding_boxes(
            bounding_boxes,
            min_size=self.min_size,
            bounding_box_format=self.bounding_box_format,
            images=images,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        return super()._batch_augment(inputs)

    def _validate_inputs(self, inputs):
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        if bounding_boxes is None:
            raise ValueError(
                "SanitizeBoundingBox expects `bounding_boxes` to be present "
                "in its inputs. For example, "
                "`layer({'images': images, 'bounding_boxes': bounding_boxes})`."
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_size": self.min_size,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config

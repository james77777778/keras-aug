from keras_cv.layers import Mosaic
from keras_cv.layers import Resizing
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from tensorflow import keras

from keras_aug.utils.augmentation_utils import BATCHED
from keras_aug.utils.augmentation_utils import BOUNDING_BOXES
from keras_aug.utils.augmentation_utils import IMAGES
from keras_aug.utils.augmentation_utils import LABELS


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
        target_size: A tuple representing the output size of images.
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
        target_size,
        offset=(0.25, 0.75),
        padding_value=114,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.resize = Resizing(
            height=target_size[0],
            width=target_size[1],
            pad_to_aspect_ratio=True,
            bounding_box_format=bounding_box_format,
            seed=seed,
        )
        self.mosaic = Mosaic(
            offset=offset,
            bounding_box_format=bounding_box_format,
            seed=seed,
        )

        self.target_size = target_size
        self.offset = offset
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def _batch_augment(self, inputs):
        outputs = self.resize(inputs)
        outputs = self.mosaic(outputs)
        return outputs

    def call(self, inputs):
        _, metadata = self._format_inputs(inputs)
        if metadata[BATCHED] is not True:
            raise ValueError(
                "Mosaic received a single image to `call`. The "
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

    def get_config(self):
        config = {
            "target_size": self.target_size,
            "offset": self.offset,
            "bounding_box_format": self.bounding_box_format,
            "seed": self.seed,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

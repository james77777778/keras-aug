import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_data_format


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomEqualize(VisionRandomLayer):
    """Equalize the histogram of the images randomly with a given probability.

    This class equalizes the histogram of the images by applying a non-linear
    mapping in order to create a uniform distribution of grayscale values in
    the outputs.

    Args:
        bins: The number of bins to use in histogram equalization. The value
            must be in the range of `[0, 256]`. Defaults to `256`.
        p: A float specifying the probability. Defaults to `0.5`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(self, bins=256, p: float = 0.5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.p = float(p)
        self.data_format = standardize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        p = ops.random.uniform([batch_size], seed=random_generator)
        return p

    def augment_images(self, images, transformations=None, **kwargs):
        ops = self.backend
        p = transformations

        def equalize(images):
            original_dtype = backend.standardize_dtype(images.dtype)
            images = self.image_backend.transform_dtype(images, "uint8")
            images = self.image_backend.equalize(
                images, self.bins, self.data_format
            )
            images = self.image_backend.transform_dtype(images, original_dtype)
            return images

        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(prob, equalize(images), images)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def _equalize_single_image(self, image):
        ops = self.backend
        if self.data_format == "channels_last":
            return ops.numpy.stack(
                [
                    self._scale_channel(image[..., c])
                    for c in range(image.shape[-1])
                ],
                axis=-1,
            )
        else:
            return ops.numpy.stack(
                [self._scale_channel(image[c]) for c in range(image.shape[-3])],
                axis=-3,
            )

    def _scale_channel(self, image_channel):
        ops = self.backend
        hist = ops.numpy.bincount(
            ops.numpy.reshape(image_channel, [-1]), minlength=self.bins
        )
        nonzero = ops.numpy.where(ops.numpy.not_equal(hist, 0), None, None)
        nonzero_hist = ops.numpy.reshape(ops.numpy.take(hist, nonzero), [-1])
        step = ops.numpy.floor_divide(
            ops.numpy.sum(hist) - nonzero_hist[-1], 255
        )

        def step_is_0():
            return ops.cast(image_channel, "uint8")

        def step_not_0():
            lut = ops.numpy.floor_divide(
                ops.numpy.add(
                    ops.numpy.cumsum(hist), ops.numpy.floor_divide(step, 2)
                ),
                step,
            )
            lut = ops.numpy.pad(lut[:-1], [[1, 0]])
            lut = ops.numpy.clip(lut, 0, 255)
            result = ops.numpy.take(lut, ops.cast(image_channel, "int64"))
            return ops.cast(result, "uint8")

        return ops.cond(step == 0, step_is_0, step_not_0)

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p, "bins": self.bins})
        return config

import typing

import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_parameter


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class ColorJitter(VisionRandomLayer):
    """Randomly change the brightness, contrast, saturation and hue.

    The input images must be 3 channels.

    Args:
        brightness: How much to jitter brightness. The factor will be chosen
            uniformly from `[1 - brightness, 1 + brightness]` or
            `[min, max]` if given the range of brightness. Set to `None` to
            deactivate brightness jittering. Defaults to `None`.
        contrast: How much to jitter contrast. The factor will be chosen
            uniformly from `[1 - contrast, 1 + contrast]` or
            `[min, max]` if given the range of contrast. Set to `None` to
            deactivate contrast jittering. Defaults to `None`.
        saturation: How much to jitter saturation. The factor will be chosen
            uniformly from `[1 - saturation, 1 + saturation]` or
            `[min, max]` if given the range of saturation. Set to `None` to
            deactivate saturation jittering. Defaults to `None`.
        hue: How much to jitter hue. The factor will be chosen
            uniformly from `[-hue, hue]` or `[min, max]` if given the range of
            hue. Should have `0 <= hue <= 0.5` or `-0.5 <= min <= max <= 0.5`.
            Set to `None` to deactivate hue jittering. Defaults to `None`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        brightness: typing.Union[None, float, typing.Sequence[float]] = None,
        contrast: typing.Union[None, float, typing.Sequence[float]] = None,
        saturation: typing.Union[None, float, typing.Sequence[float]] = None,
        hue: typing.Union[None, float, typing.Sequence[float]] = None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.brightness = standardize_parameter(
            brightness, "brightness", center=1.0, bound=(0, float("inf"))
        )
        self.contrast = standardize_parameter(
            contrast, "contrast", center=1.0, bound=(0, float("inf"))
        )
        self.saturation = standardize_parameter(
            saturation, "saturation", center=1.0, bound=(0, float("inf"))
        )
        self.hue = standardize_parameter(
            hue, "hue", center=0.0, bound=(-0.5, 0.5)
        )
        self.data_format = data_format or keras.config.image_data_format()

        if self.brightness == (0.0, 0.0):
            self.brightness = None
        if self.contrast == (0.0, 0.0):
            self.contrast = None
        if self.saturation == (0.0, 0.0):
            self.saturation = None
        if self.hue == (0.0, 0.0):
            self.hue = None

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        fn_idx = ops.random.shuffle(
            ops.numpy.arange(4, dtype="int32"), seed=random_generator
        )
        b, c, s, h = None, None, None, None

        def generate_params(low, high):
            return ops.random.uniform(
                [batch_size, 1, 1, 1], low, high, seed=random_generator
            )

        if self.brightness is not None:
            b = generate_params(self.brightness[0], self.brightness[1])
        if self.contrast is not None:
            c = generate_params(self.contrast[0], self.contrast[1])
        if self.saturation is not None:
            s = generate_params(self.saturation[0], self.saturation[1])
        if self.hue is not None:
            h = generate_params(self.hue[0], self.hue[1])

        return dict(
            fn_idx=fn_idx,
            brightness_factor=b,
            contrast_factor=c,
            saturation_factor=s,
            hue_factor=h,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations=None, **kwargs):
        ops = self.backend

        fn_idx = transformations["fn_idx"]
        brightness_factor = transformations["brightness_factor"]
        contrast_factor = transformations["contrast_factor"]
        saturation_factor = transformations["saturation_factor"]
        hue_factor = transformations["hue_factor"]

        def adjust_brightness(images):
            if self.brightness is None:
                return images
            return self.image_backend.adjust_brightness(
                images, brightness_factor
            )

        def adjust_contrast(images):
            if self.contrast is None:
                return images
            return self.image_backend.adjust_contrast(
                images,
                contrast_factor,
                data_format=self.data_format,
            )

        def adjust_saturation(images):
            if self.saturation is None:
                return images
            return self.image_backend.adjust_saturation(
                images,
                saturation_factor,
                data_format=self.data_format,
            )

        def adjust_hue(images):
            if self.hue is None:
                return images
            images = self.image_backend.adjust_hue(
                images, hue_factor, data_format=self.data_format
            )
            return images

        branches = [
            adjust_brightness,
            adjust_contrast,
            adjust_saturation,
            adjust_hue,
        ]
        for i in range(4):
            idx = fn_idx[i]
            images = ops.core.switch(idx, branches, images)
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "brightness": self.brightness,
                "contrast": self.contrast,
                "saturation": self.saturation,
                "hue": self.hue,
            }
        )
        return config

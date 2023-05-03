from functools import partial

import tensorflow as tf
from keras_cv.core import NormalFactorSampler
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug import layers
from keras_aug.core import SignedNormalFactorSampler
from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils.augmentation import IMAGES


@keras.utils.register_keras_serializable(package="keras_aug")
class RandAugment(VectorizedBaseRandomLayer):
    """RandAugment performs the Rand Augment operation on input images.

    RandAugment can be thought of as an all-in-one image augmentation layer. The
    policy implemented by RandAugment has been benchmarked extensively and is
    effective on a wide variety of datasets.

    The input images will be converted to the range [0, 255], performed
    RandAugment and then converted back to the original value range.

    For object detection tasks, you should set ``fill_mode="constant"`` and
    ``fill_value=128`` to avoid artifacts. Moreover, you can set
    ``use_geometry=False`` to turn off all geometric augmentations if the
    distortion of the bounding boxes is too large.

    Args:
        value_range ((int|float, int|float)): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        augmentations_per_image (int, optional): The number of layers to use in
            the rand augment policy. Defaults to ``2``.
        magnitude (float, optional): The shared magnitude across all
            augmentation operations. Represented as M in the paper. Usually best
            values are in the range ``[5, 10]``. Defaults to ``10``.
        magnitude_stddev (float, optional): The randomness of the severity as
            proposed by the authors of the timm library. Defaults to ``0``. When
            enabled, A gaussian noise with ``magnitude_stddev`` as sigma will be
            added to ``magnitude``.
        cutout_multiplier (float, optional): The multiplier for applying cutout.
            Defaults to ``60.0 / 331.0`` which is for ImageNet classification
            model. Usually best value is about ``0.18``.
        translation_multiplier (float, optional): The multiplier for applying
            translation. Defaults to ``150.0 / 331.0`` which is for ImageNet
            classification model. For CIFAR, it is set to ``10.0 / 32.0``.
            Usually best value is in the range ``[1.0 / 3.0, 1.0 / 2.0]``.
        use_geometry (bool, optional): whether to include geometric
            augmentations. This should be set to ``False`` when performing
            object detection. Defaults to ``True``.
        interpolation (str, optional): The interpolation mode. Supported values:
            ``"nearest", "bilinear"``. Defaults to `"nearest"`.
        fill_mode (str, optional): The fill mode. Supported values:
            ``"constant", "reflect", "wrap", "nearest"``. Defaults to
            ``"reflect"``.
        fill_value (int|float, optional): The value to be filled outside the
            boundaries when ``fill_mode="constant"``. Defaults to ``0``.
        exclude_ops (list(str), optional): Exclude selected operations.
            Defaults to ``None``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `RandAugment <https://arxiv.org/abs/1909.13719>`_
        - `Tensorflow Model augment <https://github.com/tensorflow/models/blob/v2.12.0/official/vision/ops/augment.py>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
        augmentations_per_image=2,
        magnitude=10,
        magnitude_stddev=0.0,
        cutout_multiplier=60.0 / 331.0,
        translation_multiplier=150.0 / 331.0,
        use_geometry=True,
        interpolation="nearest",
        fill_mode="reflect",
        fill_value=0,
        exclude_ops=None,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.augmentations_per_image = augmentations_per_image
        self.magnitude = magnitude
        self.magnitude_stddev = magnitude_stddev
        self.cutout_multiplier = cutout_multiplier
        self.translation_multiplier = translation_multiplier
        self.use_geometry = use_geometry
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.exclude_ops = exclude_ops
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.aug_layers = self.get_standard_policy(
            magnitude,
            magnitude_stddev,
            translation_multiplier,
            cutout_multiplier,
            use_geometry,
            exclude_ops,
            bounding_box_format,
            seed,
            **kwargs,
        )
        self.num_layers = len(self.aug_layers)

    def get_random_transformation_batch(self, batch_size):
        random_indices = self._random_generator.random_uniform(
            shape=(
                batch_size,
                self.augmentations_per_image,
            ),
            minval=0,
            maxval=self.num_layers,
            dtype=tf.int32,
        )
        return random_indices

    def _batch_augment(self, inputs):
        images = inputs.get(augmentation_utils.IMAGES, None)
        batch_size = tf.shape(images)[0]
        transformations = self.get_random_transformation_batch(batch_size)

        # images value_range transform to [0, 255]
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, (0, 255), self.compute_dtype
        )
        inputs[IMAGES] = images

        inputs_for_rand_augment_single_input = {
            "inputs": inputs,
            "transformations": transformations,
        }
        result = tf.map_fn(
            self.rand_augment_single_input,
            inputs_for_rand_augment_single_input,
        )
        # unpack result to normal inputs
        result = result["inputs"]

        # recover value_range
        images = result.get(IMAGES, None)
        images = preprocessing_utils.transform_value_range(
            images, (0, 255), self.value_range, self.compute_dtype
        )
        result[IMAGES] = images
        return result

    def get_standard_policy(
        self,
        magnitude,
        magnitude_stddev,
        translation_multiplier,
        cutout_multiplier,
        use_geometry,
        exclude_ops,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        policy = create_rand_augment_policy(
            magnitude,
            magnitude_stddev,
            translation_multiplier,
            cutout_multiplier,
        )
        aug_layers = []
        if exclude_ops is not None:
            for op in exclude_ops:
                policy.pop(op)
        for key in policy.keys():
            if key == "identity":
                aug_layers.append(layers.Identity(**policy[key], **kwargs))
            elif key == "auto_contrast":
                aug_layers.append(
                    layers.AutoContrast(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "equalize":
                aug_layers.append(
                    layers.Equalize(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "invert":
                aug_layers.append(
                    layers.Invert(**policy[key], value_range=(0, 255), **kwargs)
                )
            elif key == "posterize":
                aug_layers.append(
                    layers.RandomPosterize(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "solarize":
                aug_layers.append(
                    layers.RandomSolarize(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "color":
                aug_layers.append(
                    layers.RandomColorJitter(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "contrast":
                aug_layers.append(
                    layers.RandomColorJitter(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "brightness":
                aug_layers.append(
                    layers.RandomColorJitter(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "sharpness":
                aug_layers.append(
                    layers.RandomSharpness(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "cutout":
                aug_layers.append(
                    layers.RandomCutout(
                        **policy[key],
                        fill_mode="constant",
                        fill_value=self.fill_value,
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "solarize_add":
                aug_layers.append(
                    layers.RandomSolarize(
                        **policy[key], value_range=(0, 255), seed=seed, **kwargs
                    )
                )
            elif key == "rotate":
                if use_geometry:
                    aug_layers.append(
                        layers.RandomAffine(
                            **policy[key],
                            interpolation=self.interpolation,
                            fill_mode=self.fill_mode,
                            fill_value=self.fill_value,
                            bounding_box_format=bounding_box_format,
                            seed=seed,
                            **kwargs,
                        )
                    )
            elif key == "shear_x":
                if use_geometry:
                    aug_layers.append(
                        layers.RandomAffine(
                            **policy[key],
                            interpolation=self.interpolation,
                            fill_mode=self.fill_mode,
                            fill_value=self.fill_value,
                            bounding_box_format=bounding_box_format,
                            seed=seed,
                            **kwargs,
                        )
                    )
            elif key == "shear_y":
                if use_geometry:
                    aug_layers.append(
                        layers.RandomAffine(
                            **policy[key],
                            interpolation=self.interpolation,
                            fill_mode=self.fill_mode,
                            fill_value=self.fill_value,
                            bounding_box_format=bounding_box_format,
                            seed=seed,
                            **kwargs,
                        )
                    )
            elif key == "translate_x":
                if use_geometry:
                    aug_layers.append(
                        layers.RandomAffine(
                            **policy[key],
                            interpolation=self.interpolation,
                            fill_mode=self.fill_mode,
                            fill_value=self.fill_value,
                            bounding_box_format=bounding_box_format,
                            seed=seed,
                            **kwargs,
                        )
                    )
            elif key == "translate_y":
                if use_geometry:
                    aug_layers.append(
                        layers.RandomAffine(
                            **policy[key],
                            interpolation=self.interpolation,
                            fill_mode=self.fill_mode,
                            fill_value=self.fill_value,
                            bounding_box_format=bounding_box_format,
                            seed=seed,
                            **kwargs,
                        )
                    )
            else:
                if key not in exclude_ops:
                    raise ValueError(f"Not recognized policy key: {key}")
        return aug_layers

    def rand_augment_single_input(self, inputs):
        input = inputs.get("inputs")
        random_indices = inputs.get("transformations")

        for random_indice in random_indices:
            # construct branch_fns
            branch_fns = {}
            for idx, layer in enumerate(self.aug_layers):
                branch_fns[idx] = partial(layer, input)
            # augment
            input = tf.switch_case(random_indice, branch_fns=branch_fns)
        return {"inputs": input, "transformations": random_indices}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "augmentations_per_image": self.augmentations_per_image,
                "magnitude": self.magnitude,
                "magnitude_stddev": self.magnitude_stddev,
                "cutout_multiplier": self.cutout_multiplier,
                "translation_multiplier": self.translation_multiplier,
                "use_geometry": self.use_geometry,
                "interpolation": self.interpolation,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "exclude_ops": self.exclude_ops,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_rand_augment_policy(
    magnitude, magnitude_stddev, translation_multiplier, cutout_multiplier
):
    """Create RandAugment Policy.

    Notes:
        This policy adopts relative translatation instead of pixel adjustment.
        See discussion below:
        https://github.com/tensorflow/tpu/issues/637#issuecomment-568093430
        https://github.com/tensorflow/tpu/issues/637#issuecomment-571286096
        Author: image_size / 3.
        The cutout multiplier is about ``0.18``.
    """
    max_level = 10.0
    max_magnitude = 30.0  # AA: 10.0; RA: 30.0

    policy = {}
    policy["identity"] = {}
    policy["auto_contrast"] = {}
    policy["equalize"] = {}
    policy["invert"] = {}
    policy["rotate"] = {
        "rotation_factor": SignedNormalFactorSampler(
            mean=(magnitude / max_level) * 30.0,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=(max_magnitude / max_level) * 30.0,
        ),
        "translation_height_factor": 0,
        "translation_width_factor": 0,
        "zoom_height_factor": 0,
        "zoom_width_factor": 0,
        "shear_height_factor": 0,
        "shear_width_factor": 0,
    }
    policy["posterize"] = {
        "factor": NormalFactorSampler(
            mean=int(round(magnitude / max_level * 4)),  # must be int
            stddev=magnitude_stddev,
            min_value=0,
            max_value=8,
        )
    }
    policy["solarize"] = {
        "threshold_factor": NormalFactorSampler(
            mean=magnitude / max_level * 256,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=256,
        ),
        "addition_factor": 0,
    }
    policy["color"] = {
        "brightness_factor": 0,
        "contrast_factor": 0,
        "saturation_factor": NormalFactorSampler(
            mean=magnitude / max_level * 1.8 + 0.1,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 1.8 + 0.1,
        ),
        "hue_factor": 0,
    }
    policy["contrast"] = {
        "brightness_factor": 0,
        "contrast_factor": NormalFactorSampler(
            mean=magnitude / max_level * 1.8 + 0.1,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 1.8 + 0.1,
        ),
        "saturation_factor": 0,
        "hue_factor": 0,
    }
    policy["brightness"] = {
        "brightness_factor": NormalFactorSampler(
            mean=magnitude / max_level * 1.8 + 0.1,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 1.8 + 0.1,
        ),
        "contrast_factor": 0,
        "saturation_factor": 0,
        "hue_factor": 0,
    }
    policy["sharpness"] = {
        "factor": NormalFactorSampler(
            mean=magnitude / max_level * 1.8 + 0.1,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 1.8 + 0.1,
        )
    }
    policy["shear_x"] = {
        "rotation_factor": 0,
        "translation_height_factor": 0,
        "translation_width_factor": 0,
        "zoom_height_factor": 0,
        "zoom_width_factor": 0,
        "shear_height_factor": 0,
        "shear_width_factor": SignedNormalFactorSampler(
            mean=magnitude / max_level * 0.3,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 0.3,
            rate=0.5,
        ),
    }
    policy["shear_y"] = {
        "rotation_factor": 0,
        "translation_height_factor": 0,
        "translation_width_factor": 0,
        "zoom_height_factor": 0,
        "zoom_width_factor": 0,
        "shear_height_factor": SignedNormalFactorSampler(
            mean=magnitude / max_level * 0.3,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 0.3,
            rate=0.5,
        ),
        "shear_width_factor": 0,
    }
    policy["translate_x"] = {
        "rotation_factor": 0,
        "translation_height_factor": 0,
        "translation_width_factor": 0,
        "zoom_height_factor": 0,
        "zoom_width_factor": 0,
        "shear_height_factor": 0,
        "shear_width_factor": SignedNormalFactorSampler(
            mean=magnitude / max_level * translation_multiplier,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * translation_multiplier,
            rate=0.5,
        ),
    }
    policy["translate_y"] = {
        "rotation_factor": 0,
        "translation_height_factor": 0,
        "translation_width_factor": 0,
        "zoom_height_factor": 0,
        "zoom_width_factor": 0,
        "shear_height_factor": SignedNormalFactorSampler(
            mean=magnitude / max_level * translation_multiplier,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * translation_multiplier,
            rate=0.5,
        ),
        "shear_width_factor": 0,
    }
    policy["cutout"] = {
        "height_factor": NormalFactorSampler(
            mean=magnitude / max_level * cutout_multiplier,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * cutout_multiplier,
        ),
        "width_factor": NormalFactorSampler(
            mean=magnitude / max_level * cutout_multiplier,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * cutout_multiplier,
        ),
    }
    policy["solarize_add"] = {
        "threshold_factor": (128, 128),
        "addition_factor": NormalFactorSampler(
            mean=magnitude / max_level * 110,
            stddev=magnitude_stddev,
            min_value=0,
            max_value=max_magnitude / max_level * 110,
        ),
    }
    return policy
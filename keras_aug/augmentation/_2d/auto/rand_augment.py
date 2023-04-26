from functools import partial

import tensorflow as tf
from keras_cv.core import NormalFactorSampler
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug import augmentation
from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.core import SignedNormalFactorSampler
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
            Defaults to ``40``.
        translation_multiplier (float, optional): The multiplier for applying
            translation. Defaults to ``150.0 / 331.0`` which is for ImageNet
            classification model. For CIFAR, it is set to ``10.0 / 32.0``.
            Usually best values are in the range ``[1.0 / 3.0, 1.0 / 2.0]``.
        fill_value (int|float, optional): The value to be filled outside the
            boundaries when ``fill_mode="constant"``. Defaults to ``0``.
        use_geometry (bool, optional): whether to include geometric
            augmentations. This should be set to ``False`` when performing
            object detection. Defaults to ``True``.
        batchwise (bool, optional): whether to run RandAugment in batchwise.
            When enabled, RandAugment might run faster by truely vectorizing
            the augmentations.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `RandAugment <https://arxiv.org/abs/1909.13719>`_
        - `Tensorflow Model <https://github.com/tensorflow/models/blob/v2.12.0/official/vision/ops/augment.py>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
        augmentations_per_image=2,
        magnitude=10,
        magnitude_stddev=0.0,
        cutout_multiplier=40.0,
        translation_multiplier=150.0 / 331.0,
        fill_value=0,
        use_geometry=True,
        batchwise=False,
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
        self.fill_value = fill_value
        self.use_geometry = use_geometry
        self.batchwise = batchwise
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.layers = self.get_standard_policy(
            magnitude,
            magnitude_stddev,
            translation_multiplier,
            use_geometry,
            bounding_box_format,
            seed,
            **kwargs,
        )
        self.num_layers = len(self.layers)

    def get_random_transformation_batch(self, batch_size):
        random_indices = self._random_generator.random_uniform(
            shape=(
                batch_size if not self.batchwise else 1,
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

        if self.batchwise:
            result = inputs.copy()
            for random_indice in transformations[0]:
                # construct branch_fns
                branch_fns = {}
                for idx, layer in enumerate(self.layers):
                    branch_fns[idx] = partial(layer, result)
                # augment
                result = tf.switch_case(random_indice, branch_fns=branch_fns)
        else:
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
        use_geometry,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        policy = create_rand_augment_policy(
            magnitude, magnitude_stddev, translation_multiplier
        )

        identity = augmentation.Identity(**policy["identity"], **kwargs)
        auto_contrast = augmentation.AutoContrast(
            **policy["auto_contrast"],
            value_range=(0, 255),
            seed=seed,
            **kwargs,
        )
        equalize = augmentation.Equalize(
            **policy["equalize"], value_range=(0, 255), seed=seed, **kwargs
        )
        invert = augmentation.Invert(
            **policy["invert"], value_range=(0, 255), **kwargs
        )
        posterize = augmentation.RandomPosterize(
            **policy["posterize"], value_range=(0, 255), seed=seed, **kwargs
        )
        solarize = augmentation.RandomSolarize(
            **policy["solarize"], value_range=(0, 255), seed=seed, **kwargs
        )
        color = augmentation.RandomColorJitter(
            **policy["color"], value_range=(0, 255), seed=seed, **kwargs
        )
        contrast = augmentation.RandomColorJitter(
            **policy["contrast"], value_range=(0, 255), seed=seed, **kwargs
        )
        brightness = augmentation.RandomColorJitter(
            **policy["brightness"], value_range=(0, 255), seed=seed, **kwargs
        )
        sharpness = augmentation.RandomSharpness(
            **policy["sharpness"], value_range=(0, 255), seed=seed, **kwargs
        )
        # TODO: CutOut layer
        solarize_add = augmentation.RandomSolarize(
            **policy["solarize_add"],
            value_range=(0, 255),
            seed=seed,
            **kwargs,
        )
        layers = [
            identity,
            auto_contrast,
            equalize,
            invert,
            posterize,
            solarize,
            color,
            contrast,
            brightness,
            sharpness,
            solarize_add,
        ]

        if use_geometry:
            rotate = augmentation.RandomAffine(
                **policy["rotate"],
                fill_value=self.fill_value,
                bounding_box_format=bounding_box_format,
                seed=seed,
                **kwargs,
            )
            shear_x = augmentation.RandomAffine(
                **policy["shear_x"],
                fill_value=self.fill_value,
                bounding_box_format=bounding_box_format,
                seed=seed,
                **kwargs,
            )
            shear_y = augmentation.RandomAffine(
                **policy["shear_y"],
                fill_value=self.fill_value,
                bounding_box_format=bounding_box_format,
                seed=seed,
                **kwargs,
            )
            translate_x = augmentation.RandomAffine(
                **policy["translate_x"],
                fill_value=self.fill_value,
                bounding_box_format=bounding_box_format,
                seed=seed,
                **kwargs,
            )
            translate_y = augmentation.RandomAffine(
                **policy["translate_y"],
                fill_value=self.fill_value,
                bounding_box_format=bounding_box_format,
                seed=seed,
                **kwargs,
            )
            layers.extend([rotate, shear_x, shear_y, translate_x, translate_y])
        return layers

    def rand_augment_single_input(self, inputs):
        input = inputs.get("inputs")
        random_indices = inputs.get("transformations")

        for random_indice in random_indices:
            # construct branch_fns
            branch_fns = {}
            for idx, layer in enumerate(self.layers):
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
                "fill_value": self.fill_value,
                "use_geometry": self.use_geometry,
                "batchwise": self.batchwise,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_rand_augment_policy(
    magnitude, magnitude_stddev, translation_multiplier
):
    """Create RandAugment Policy.

    TODO: CutOut

    Notes:
        This policy adopts relative translatation instead of pixel adjustment.
        See discussion below:
        https://github.com/tensorflow/tpu/issues/637#issuecomment-568093430
        https://github.com/tensorflow/tpu/issues/637#issuecomment-571286096
        Author: image_size / 3

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
    policy["cutout"] = {}  # TODO
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

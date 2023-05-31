from functools import partial

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug import layers
from keras_aug.core import UniformFactorSampler
from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES


@keras.utils.register_keras_serializable(package="keras_aug")
class TrivialAugmentWide(VectorizedBaseRandomLayer):
    """TrivialAugmentWide performs the Wide version of Trivial Augment
    operation on input images.

    TrivialAugmentWide can be thought of as an all-in-one image augmentation
    layer. The policy implemented by TrivialAugmentWide has been benchmarked
    extensively and is effective on a wide variety of datasets.

    The input images will be converted to the range [0, 255], performed
    TrivialAugment and then converted back to the original value range.

    For object detection tasks, you should set ``fill_mode="constant"`` and
    ``fill_value=128`` to avoid artifacts. Moreover, you can set
    ``use_geometry=False`` to turn off all geometric augmentations if the
    distortion of the bounding boxes is too large.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
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
        - `TrivialAugment <https://arxiv.org/abs/2103.10158>`_
        - `TrivialAugment Official Repo <https://github.com/automl/trivialaugment>`_
        - `torchvision <https://github.com/pytorch/vision>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
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
        self.use_geometry = use_geometry
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.exclude_ops = exclude_ops
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.aug_layers = self.get_standard_policy(
            use_geometry,
            exclude_ops,
            bounding_box_format,
            seed,
            **kwargs,
        )
        self.num_layers = len(self.aug_layers)

    def get_standard_policy(
        self,
        use_geometry,
        exclude_ops,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        policy = create_trivial_augment_policy(seed)
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
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "equalize":
                aug_layers.append(
                    layers.Equalize(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "posterize":
                aug_layers.append(
                    layers.RandomPosterize(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "solarize":
                aug_layers.append(
                    layers.RandomSolarize(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "color":
                aug_layers.append(
                    layers.RandomColorJitter(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "contrast":
                aug_layers.append(
                    layers.RandomColorJitter(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "brightness":
                aug_layers.append(
                    layers.RandomColorJitter(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
                    )
                )
            elif key == "sharpness":
                aug_layers.append(
                    layers.RandomSharpness(
                        **policy[key],
                        value_range=(0, 255),
                        seed=seed,
                        **kwargs,
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
                raise ValueError(f"Not recognized policy key: {key}")
        return aug_layers

    def get_random_transformation_batch(self, batch_size):
        random_indices = self._random_generator.random_uniform(
            shape=(batch_size,),
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
            images, self.value_range, (0, 255), dtype=self.compute_dtype
        )
        inputs[IMAGES] = images

        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        # make bounding_boxes to dense first
        if bounding_boxes is not None:
            ori_bbox_info = bounding_box.validate_format(bounding_boxes)
            inputs[BOUNDING_BOXES] = bounding_box.to_dense(bounding_boxes)

        inputs_for_trivial_augment_single_input = {
            "inputs": inputs,
            "transformations": transformations,
        }
        result = tf.map_fn(
            self.trivial_augment_single_input,
            inputs_for_trivial_augment_single_input,
            fn_output_signature=augmentation_utils.compute_signature(
                inputs, self.compute_dtype
            ),
        )

        bounding_boxes = result.get(BOUNDING_BOXES, None)
        if bounding_boxes is not None:
            if ori_bbox_info["ragged"]:
                bounding_boxes = bounding_box.to_ragged(bounding_boxes)
            else:
                bounding_boxes = bounding_box.to_dense(bounding_boxes)
            result[BOUNDING_BOXES] = bounding_boxes

        # recover value_range
        images = result.get(IMAGES, None)
        images = preprocessing_utils.transform_value_range(
            images, (0, 255), self.value_range, self.compute_dtype
        )
        result[IMAGES] = images
        return result

    def trivial_augment_single_input(self, inputs):
        input = inputs.get("inputs")
        random_indice = inputs.get("transformations")
        # construct branch_fns
        branch_fns = {}
        for idx, layer in enumerate(self.aug_layers):
            branch_fns[idx] = partial(layer, input)

        # TODO:
        # figure out why tf will make tf.float32 instead of tf.float16
        # keras.mixed_precision.set_global_policy("mixed_float16")
        result = tf.switch_case(random_indice, branch_fns=branch_fns)
        if BOUNDING_BOXES in result:
            result[BOUNDING_BOXES] = bounding_box.to_ragged(
                result[BOUNDING_BOXES], dtype=self.compute_dtype
            )
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
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


def create_trivial_augment_policy(seed):
    """Create TrivialAugment Policy.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_auto_augment.py
    """  # noqa: E501
    policy = {}
    policy["identity"] = {}
    policy["auto_contrast"] = {}
    policy["equalize"] = {}
    policy["rotate"] = {
        "rotation_factor": UniformFactorSampler(
            lower=-135, upper=135, seed=seed
        ),
    }
    policy["posterize"] = {
        "factor": UniformFactorSampler(lower=2, upper=8 + 1, seed=seed)
    }
    policy["solarize"] = {
        "threshold_factor": UniformFactorSampler(lower=0, upper=255, seed=seed),
        "addition_factor": 0,
    }
    policy["color"] = {
        "saturation_factor": UniformFactorSampler(
            lower=1, upper=1.99, seed=seed
        ),
    }
    policy["contrast"] = {
        "contrast_factor": UniformFactorSampler(lower=1, upper=1.99, seed=seed),
    }
    policy["brightness"] = {
        "brightness_factor": UniformFactorSampler(
            lower=1, upper=1.99, seed=seed
        ),
    }
    policy["sharpness"] = {
        "factor": UniformFactorSampler(lower=1, upper=1.99, seed=seed),
    }
    policy["shear_x"] = {
        "shear_height_factor": 0,
        "shear_width_factor": UniformFactorSampler(
            lower=-0.99, upper=0.99, seed=seed
        ),
    }
    policy["shear_y"] = {
        "shear_height_factor": UniformFactorSampler(
            lower=-0.99, upper=0.99, seed=seed
        ),
        "shear_width_factor": 0,
    }
    policy["translate_x"] = {
        "translation_height_factor": 0,
        "translation_width_factor": UniformFactorSampler(
            lower=-1, upper=1, seed=seed
        ),
    }
    policy["translate_y"] = {
        "translation_height_factor": UniformFactorSampler(
            lower=-1, upper=1, seed=seed
        ),
        "translation_width_factor": 0,
    }
    return policy

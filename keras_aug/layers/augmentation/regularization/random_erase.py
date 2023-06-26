import tensorflow as tf
from keras_cv.utils import fill_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomErase(VectorizedBaseRandomLayer):
    """Randomly erase rectangles from images and fill them.

    Args:
        area_factor (float|Sequence[float]|keras_aug.FactorSampler, optional): The range
            of the area factor that controls the area of the erasing. When
            represented as a single float, the factor will be picked between
            ``[0.0, 0.0 + upper]``. ``0.0`` means the rectangle will be of size
            0% of the image area. ``0.1`` means the rectangle will have a size
            of 10% of the image area. Defaults to ``(0.02, 0.4)``
        aspect_ratio_factor (float|Sequence[float]|keras_aug.FactorSampler, optional): The
            range of the aspect ratio factor that controls the aspect ratio of
            the erasing. When represented as a single float, the factor will be
            picked between ``[1.0 - lower, 1.0 + upper]``. ``1.0`` means the
            erasing will be square. Defaults to ``(0.3, 1.0 / 0.3)``.
        fill_mode (str, optional): Pixels inside the erasing are filled
            according to the given mode. Supported values:
            ``"constant", "gaussian_noise"``. Defaults to ``"constant"``.
        fill_value (tuple(float), optional): The values to be filled in the
            erasing when ``fill_mode="constant"``. Defaults to
            ``(125, 123, 114)`` which is the means of ImageNet.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `Random Erasing <https://arxiv.org/abs/1708.04896v2>`_
        - `Random Erasing Official Repo <https://github.com/zhunzhong07/Random-Erasing>`_
        - `Tensorflow Model augment <https://github.com/tensorflow/models/blob/v2.12.0/official/vision/ops/augment.py>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        area_factor=(0.02, 0.4),
        aspect_ratio_factor=(0.3, 1.0 / 0.3),
        fill_mode="constant",
        fill_value=(125, 123, 114),
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(area_factor, (int, float)):
            area_factor = (0, area_factor)
        self.area_factor = augmentation_utils.parse_factor(
            area_factor, seed=seed
        )
        if isinstance(aspect_ratio_factor, (int, float)):
            aspect_ratio_factor = (
                1.0 - aspect_ratio_factor,
                1.0 + aspect_ratio_factor,
            )
        self.aspect_ratio_factor = augmentation_utils.parse_factor(
            aspect_ratio_factor, max_value=None, seed=seed
        )
        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant". Got `fill_mode`={fill_mode}'
            )
        self.fill_mode = fill_mode
        self.fill_value = tf.convert_to_tensor(
            fill_value, dtype=self.compute_dtype
        )
        self.seed = seed

        self.max_attemp = 10

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        areas = heights * widths

        is_success = False
        for _ in range(self.max_attemp):
            if not is_success:
                erasing_areas = self.area_factor(shape=(batch_size, 1))
                erasing_areas = erasing_areas * areas
                erasing_aspect_ratios = self.aspect_ratio_factor(
                    shape=(batch_size, 1)
                )
                erasing_heights = tf.round(
                    tf.sqrt(erasing_areas * erasing_aspect_ratios)
                )
                erasing_widths = tf.round(
                    tf.sqrt(erasing_areas / erasing_aspect_ratios)
                )
                if tf.reduce_all(
                    tf.less(erasing_widths, widths)
                ) and tf.reduce_all(tf.less(erasing_heights, heights)):
                    is_success = True

        center_xs = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        center_ys = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        center_xs = tf.round(center_xs * (widths - erasing_widths))
        center_xs = tf.cast(center_xs + erasing_widths / 2, dtype=tf.int32)
        center_ys = tf.round(center_ys * (heights - erasing_heights))
        center_ys = tf.cast(center_ys + erasing_heights / 2, dtype=tf.int32)

        return {
            "center_xs": center_xs,
            "center_ys": center_ys,
            "erasing_heights": tf.cast(erasing_heights, dtype=tf.int32),
            "erasing_widths": tf.cast(erasing_widths, dtype=tf.int32),
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = tf.cast(images, dtype=self.compute_dtype)
        center_xs = transformations["center_xs"]
        center_ys = transformations["center_ys"]
        erasing_heights = transformations["erasing_heights"]
        erasing_widths = transformations["erasing_widths"]
        rectangle_fills = self.compute_rectangle_fills(images)
        images = fill_utils.fill_rectangle(
            images,
            center_xs[..., 0],
            center_ys[..., 0],
            erasing_widths[..., 0],
            erasing_heights[..., 0],
            rectangle_fills,
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def compute_rectangle_fills(self, inputs):
        if self.fill_mode == "constant":
            fills = tf.ones(tf.shape(inputs), dtype=self.compute_dtype)
            fills = (
                fills * self.fill_value[tf.newaxis, tf.newaxis, tf.newaxis, ...]
            )
        else:
            # gaussian noise
            fills = tf.random.normal(tf.shape(inputs), dtype=self.compute_dtype)
        return fills

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "area_factor": self.area_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "seed": self.seed,
            }
        )
        return config

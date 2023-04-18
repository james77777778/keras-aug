import tensorflow as tf
from keras_cv import core
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomJpegQuality(VectorizedBaseImageAugmentationLayer):
    """Applies Random Jpeg compression artifacts to an image.

    Performs the jpeg compression algorithm on the image. This layer can be used
    in order to ensure your model is robust to artifacts introduced by JPEG
    compression.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high]. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        factor: positive float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound. When
            represented as a single float, the factor will be randomly picked
            between `[1.0 - factor, 1.0]`. When 0.5 is chosen, the output image
            will be compressed 50% with JPEG, and when 1.0 is chosen, it is
            still lossy compresson. This value is passed to
            `tf.image.adjust_jpeg_quality()`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        value_range,
        factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(factor, float):
            factor = (1.0 - factor, 1.0)
        elif isinstance(factor, list) or isinstance(factor, tuple):
            factor = sorted(factor)
            factor = (factor[0], factor[1])
        elif isinstance(factor, core.UniformFactorSampler):
            factor = factor
        else:
            raise ValueError(
                "RandomJpeqQuality expects `factor` to be a list, a tuple of "
                f"ints or an int. Got `factor`: {factor}"
            )
        self.factor = preprocessing_utils.parse_factor(
            factor, min_value=0, max_value=1, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # scale from [0, 1] to [0, 100]
        factors = self.factor(shape=(batch_size, 1), dtype=tf.float32)
        factors = tf.cast(tf.round(factors * 100), dtype=tf.int32)
        return factors

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images,
            self.value_range,
            target_range=(0, 1),
            dtype=tf.float32,
        )

        inputs_for_adjust_jpeg_qualitye = {
            "images": images,
            "factors": transformations,
        }
        jpeq_compressed_images = tf.vectorized_map(
            self.adjust_jpeg_quality,
            inputs_for_adjust_jpeg_qualitye,
        )

        jpeq_compressed_images = preprocessing_utils.transform_value_range(
            jpeq_compressed_images,
            (0, 1),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        return jpeq_compressed_images

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

    def adjust_jpeg_quality(self, inputs):
        image = inputs.get("images", None)

        image = tf.cast(image, tf.float32)
        factor = inputs.get("factors", None)

        image = tf.image.adjust_jpeg_quality(image, jpeg_quality=factor[0])
        return image

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "factor": self.factor,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

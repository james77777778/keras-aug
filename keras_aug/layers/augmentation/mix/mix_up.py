import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import BATCHED
from keras_aug.utils.distribution import stateless_random_beta


@keras.utils.register_keras_serializable(package="keras_aug")
class MixUp(VectorizedBaseRandomLayer):
    """The MixUp data augmentation technique.

    The MixUp data augmentation technique involves taking 2 images from a given
    batch and fusing them together using a ratio sampled from a beta
    distribution. Labels are applied by same ratio ratio. Bounding boxes are
    concated according to the position of the 2 images.

    Args:
        alpha (float, optional): The inverse scale parameter between 0 to +inf
            for the gamma distribution. This controls the shape of the
            distribution from which the smoothing values are sampled. Defaults
            to ``0.2``, which is a recommended value when training an ImageNet
            classification model. For object detection, it is recommended to use
            a larger value. For example YOLOV8 uses ``32.0``.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `MixUp <https://arxiv.org/abs/1710.09412>`_
        - `Bag of Freebies for Training Object Detection Neural Networks <https://arxiv.org/abs/1902.04103>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(self, alpha=0.2, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.seed = seed

        # set force_no_unwrap_ragged_image_call=True because MixUp needs
        # to process images in batch.
        self.force_no_unwrap_ragged_image_call = True

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # cast to float32 to avoid numerical issue
        permutation_order = tf.argsort(
            self._random_generator.random_uniform((batch_size,)), axis=-1
        )
        lambda_samples = stateless_random_beta(
            (batch_size, 1, 1, 1),
            seed_alpha=self._random_generator.make_seed_for_stateless_op(),
            seed_beta=self._random_generator.make_seed_for_stateless_op(),
            alpha=self.alpha,
            beta=self.alpha,
            dtype=tf.float32,
        )
        return {
            "permutation_order": permutation_order,
            "lambda_samples": lambda_samples,
        }

    def augment_images(self, images, transformations, **kwargs):
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "MixUp expects dense images. Received: images type: "
                f"{type(images)}"
            )
        permutation_order = transformations["permutation_order"]
        lambda_samples = tf.cast(
            transformations["lambda_samples"], dtype=self.compute_dtype
        )
        mixup_images = tf.gather(images, permutation_order)
        mixup_images = tf.cast(mixup_images, dtype=self.compute_dtype)
        images = lambda_samples * images + (1.0 - lambda_samples) * mixup_images
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        permutation_order = transformations["permutation_order"]
        lambda_samples = tf.cast(
            transformations["lambda_samples"], dtype=self.compute_dtype
        )
        labels = tf.cast(labels, dtype=self.compute_dtype)
        mixup_labels = tf.gather(labels, permutation_order)
        lambda_samples = tf.reshape(lambda_samples, [-1, 1])
        labels = lambda_samples * labels + (1.0 - lambda_samples) * mixup_labels
        return labels

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        permutation_order = transformations["permutation_order"]
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]
        mixup_boxes = tf.gather(boxes, permutation_order)
        mixup_classes = tf.gather(classes, permutation_order)
        boxes = tf.concat([boxes, mixup_boxes], axis=1)
        classes = tf.concat([classes, mixup_classes], axis=1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes["classes"] = classes
        return bounding_boxes

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        return super()._batch_augment(inputs)

    def call(self, inputs):
        _, metadata = self._format_inputs(inputs)
        if metadata[BATCHED] is not True:
            raise ValueError(
                "MixUp received a single image to `call`. The "
                "layer relies on combining multiple examples, and as such "
                "will not behave as expected. Please call the layer with 2 "
                "or more samples."
            )
        return super().call(inputs=inputs)

    def _validate_inputs(self, inputs):
        images = inputs.get("images", None)
        labels = inputs.get("labels", None)
        bounding_boxes = inputs.get("bounding_boxes", None)
        if images is None or (labels is None and bounding_boxes is None):
            raise ValueError(
                "MixUp expects inputs in a dictionary with format "
                '{"images": images, "labels": labels}. or'
                '{"images": images, "bounding_boxes": bounding_boxes}'
                f"Got: inputs = {inputs}."
            )
        if bounding_boxes is not None:
            _ = bounding_box.validate_format(bounding_boxes)

    def sample_from_beta(self, alpha, beta, shape):
        sample_alpha = tf.random.stateless_gamma(
            shape,
            alpha=alpha,
            seed=self._random_generator.make_seed_for_stateless_op(),
        )
        sample_beta = tf.random.stateless_gamma(
            shape,
            alpha=beta,
            seed=self._random_generator.make_seed_for_stateless_op(),
        )
        return sample_alpha / (sample_alpha + sample_beta)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "seed": self.seed})
        return config

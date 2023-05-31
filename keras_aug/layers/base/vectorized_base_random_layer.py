import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing
from tensorflow import keras

from keras_aug.layers.__internal__.base_layer import BaseRandomLayer

H_AXIS = -3
W_AXIS = -2

IMAGES = "images"
LABELS = "labels"
TARGETS = "targets"
BOUNDING_BOXES = "bounding_boxes"
KEYPOINTS = "keypoints"
SEGMENTATION_MASKS = "segmentation_masks"
CUSTOM_ANNOTATIONS = "custom_annotations"

IS_DICT = "is_dict"
BATCHED = "batched"
USE_TARGETS = "use_targets"


@keras.utils.register_keras_serializable(package="keras_aug")
class VectorizedBaseRandomLayer(BaseRandomLayer):
    """Abstract base layer for vectorized image augmentation.

    This layer contains base functionalities for preprocessing layers which
    augment image related data, e.g. image and in the future, label and bounding
    boxes. The subclasses could avoid making certain mistakes and reduce code
    duplications.

    This layer requires you to implement one method ``augment_images()``, which
    augments one single image during the training. There are a few additional
    methods that you can implement for added functionality on the layer.

    ``augment_ragged_image()`` and ``compute_ragged_image_signature()``, which
    handles ragged images augmentation if the layer supports that.

    ``augment_labels()``, which handles label augmentation if the layer
    supports that.

    ``augment_bounding_boxes()``, which handles the bounding box
    augmentation, if the layer supports that.

    ``augment_keypoints()``, which handles the keypoints augmentation, if the
    layer supports that.

    ``augment_segmentation_masks()``, which handles the segmentation masks
    augmentation, if the layer supports that.

    ``augment_custom_annotations()``, which handles the custom annotations
    augmentation, if the layer supports that. This is useful to implement
    augmentation for special annotatinos.

    ``get_random_transformations()``, which should produce a batch of random
    transformation settings. The transformation object, which must be a
    batched Tensor or a dictionary where each input is a batched Tensor,
    will be passed to ``augment_images``, ``augment_labels`` and
    `augment_bounding_boxes`, to coordinate the randomness behavior, eg, in
    the RandomFlip layer, the image and bounding_boxes should be changed in
    the same way.

    The ``call()`` method support two formats of inputs::

        1. Single image tensor with 3D (HWC) or 4D (NHWC) format.
        2. A dict of tensors with stable keys. The supported keys are
            ``"images"``, ``"labels"``, ``"bounding_boxes"``,
            ``segmentation_masks``, ``keypoints`` and ``custom_annotations`` at
            the moment. We might add more keys in future when we support more
            types of augmentation.

    The output of the ``call()`` will be in two formats, which will be the same
    structure as the inputs.

    The ``call()`` will unpack the inputs, forward to the correct function, and
    pack the output back to the same structure as the inputs.

    By default, the dense or ragged status of the output will be preserved.
    However, you can override this behavior by setting
    ``self.force_output_dense_images = True`` in your ``__init__()``
    method. When enabled, images and segmentation masks will be converted to
    dense tensor by ``to_tensor()`` if ragged.

    .. code-block:: python

        class SubclassLayer(VectorizedBaseImageAugmentationLayer):
            def __init__(self):
                super().__init__()
                self.force_output_dense_images = True

    Note that since the randomness is also a common functionality, this layer
    also includes a keras.backend.RandomGenerator, which can be used to
    produce the random numbers. The random number generator is stored in the
    `self._random_generator` attribute.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(
        self, seed=None, force_generator=True, rng_type="stateless", **kwargs
    ):
        # Defaults to stateless random operations
        super().__init__(
            seed=seed,
            force_generator=force_generator,
            rng_type=rng_type,
            **kwargs,
        )

    @property
    def force_no_unwrap_ragged_image_call(self):
        """Control whether to force not to unwrap ragged image call."""
        return getattr(self, "_force_no_unwrap_ragged_image_call", False)

    @force_no_unwrap_ragged_image_call.setter
    def force_no_unwrap_ragged_image_call(
        self, force_no_unwrap_ragged_image_call
    ):
        self._force_no_unwrap_ragged_image_call = (
            force_no_unwrap_ragged_image_call
        )

    @property
    def force_output_dense_images(self):
        """Control whether to force outputting of dense images."""
        return getattr(self, "_force_output_dense_images", False)

    @force_output_dense_images.setter
    def force_output_dense_images(self, force_output_dense_images):
        self._force_output_dense_images = force_output_dense_images

    def get_random_transformation_batch(
        self,
        batch_size,
        images=None,
        labels=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_masks=None,
        custom_annotations=None,
    ):
        """Produce random transformations config for a batch of inputs.

        This is used to produce same randomness between image / label /
        bounding_box.

        Args:
            batch_size: the batch size of transformations configuration to
                sample.
            images: 3D image tensor from inputs.
            labels: optional 1D label tensor from inputs.
            bounding_boxes: optional 2D bounding boxes tensor from inputs.
            segmentation_masks: optional 3D segmentation mask tensor from
                inputs.

        Returns:
            Any type of object, which will be forwarded to `augment_images`,
            `augment_labels` and `augment_bounding_boxes` as the
            `transformations` parameter.
        """
        # Required to work with map_fn in the ragged cast.
        return tf.zeros((batch_size))

    def compute_ragged_image_signature(self, images):
        """Computes the output image signature for the
        `_unwrap_ragged_image_call()` function.

        Must be overridden to return tensors with different shapes than the
        input images. By default, returns either a `tf.RaggedTensorSpec`
        matching the input image spec, or a `tf.TensorSpec` matching the input
        image spec.
        """
        return tf.RaggedTensorSpec(
            shape=images.shape[1:],
            ragged_rank=1,
            dtype=self.compute_dtype,
        )

    def compute_ragged_segmentation_mask_signature(self, segmentation_maks):
        """Computes the output segmentation_mask signature for the
        `_unwrap_ragged_segmentation_call()` function.

        Must be overridden to return tensors with different shapes than the
        input images. By default, returns either a `tf.RaggedTensorSpec`
        matching the input image spec, or a `tf.TensorSpec` matching the input
        image spec.
        """
        return tf.RaggedTensorSpec(
            shape=segmentation_maks.shape[1:],
            ragged_rank=1,
            dtype=self.compute_dtype,
        )

    def augment_ragged_image(self, image, transformation, **kwargs):
        """Augment an image from a ragged image batch.

        This method accepts a single Dense image Tensor, and returns a Dense
        image. The resulting images are then stacked back into a ragged image
        batch. The behavior of this method should be identical to that of
        `augment_images()` but is to operate on a batch-wise basis.

        Args:
            image: a single image from the batch
            transformation: a single transformation sampled from
                `get_random_transformations()`.
            kwargs: all the other call arguments (i.e. bounding_boxes, labels,
                etc.).

        Returns:
            Augmented image.
        """
        raise NotImplementedError(
            "A ragged image batch was passed to layer of type "
            f"`{type(self).__name__}`. This layer does not implement "
            "`augment_ragged_image()`. If this is a `keras_aug`, open a GitHub "
            "issue requesting Ragged functionality on the layer titled: "
            f"'`{type(self).__name__}`: ragged image support'. If this is a "
            "custom layer, implement the `augment_ragged_image()` method."
        )

    def augment_images(self, images, transformations, **kwargs):
        """Augment a batch of images.

        Args:
            images: 4D image input tensor to the layer. Forwarded from
                `layer.call()`. This should generally have the shape
                [B, H, W, C]. Forwarded from `layer.call()`.
            transformations: The transformations object produced by
                `get_random_transformations`. Used to coordinate the randomness
                between image, label, bounding box, keypoints, and segmentation
                mask.

        Returns:
            output 4D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_labels(self, labels, transformations, **kwargs):
        """Augment a batch of labels.

        Args:
            labels: 2D label to the layer. Forwarded from `layer.call()`.
            transformations: The transformations object produced by
                `get_random_transformations`. Used to coordinate the randomness
                between image, label, bounding box, keypoints, and segmentation
                mask.

        Returns:
            output 2D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        """Augment bounding boxes for one image.

        Args:
            bounding_boxes: 3D bounding boxes to the layer. Forwarded from
                `call()`.
            transformations: The transformations object produced by
                `get_random_transformations`. Used to coordinate the randomness
                between image, label, bounding box, keypoints, and segmentation
                mask.

        Returns:
            output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        """Augment a batch of keypoints for one image.

        Args:
            keypoints: 3D keypoints input tensor to the layer. Forwarded from
                `layer.call()`. Shape should be [batch, num_keypoints, 2] in the
                specified keypoint format.
            transformations: The transformations object produced by
                `get_random_transformations`. Used to coordinate the randomness
                between image, label, bounding box, keypoints, and segmentation
                mask.

        Returns:
            output 3D tensor, which will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_ragged_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        """Augment an image from a ragged segmentation mask batch.

        This method accepts a single Dense image Tensor, and returns a Dense
        image. The resulting images are then stacked back into a ragged image
        batch. The behavior of this method should be identical to that of
        `augment_segmentation_masks()` but is to operate on a batch-wise basis.

        Args:
            segmentation_mask: a single image from the batch
            transformation: a single transformation sampled from
                `get_random_transformations()`.
            kwargs: all the other call arguments (i.e. bounding_boxes, labels,
                etc.).

        Returns:
            Augmented segmentation mask.
        """
        return segmentation_mask

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        """Augment a batch of images' segmentation masks.

        Args:
            segmentation_masks: 4D segmentation mask input tensor to the layer.
                This should generally have the shape [B, H, W, 1], or in some
                cases [B, H, W, C] for multilabeled data. Forwarded from
                `layer.call()`.
            transformations: The transformations object produced by
                `get_random_transformations`. Used to coordinate the randomness
                between image, label, bounding box, keypoints, and segmentation
                mask.

        Returns:
            output 4D tensor containing the augmented segmentation mask, which
            will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def augment_custom_annotations(
        self, custom_annotations, transformations, **kwargs
    ):
        """Augment a batch of images' custom_annotations.

        Args:
            custom_annotations: 4D custom annotations input tensor to the layer.
            transformations: The transformations object produced by
                `get_random_transformations`. Used to coordinate the randomness
                between image, label, bounding box, keypoints, and segmentation
                mask.

        Returns:
            output 4D tensor containing the augmented segmentation mask, which
            will be forward to `layer.call()`.
        """
        raise NotImplementedError()

    def _unwrap_ragged_image_call(self, inputs):
        images = inputs.get(IMAGES, None)
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        keypoints = inputs.get(KEYPOINTS, None)
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        transformations = inputs.get("transformations")
        images = images.to_tensor()
        images = self.augment_ragged_image(
            image=images,
            label=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_mask=segmentation_masks,
            transformation=transformations,
        )
        return tf.RaggedTensor.from_tensor(images)

    def _unwrap_ragged_segmentation_mask_call(self, inputs):
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        transformations = inputs.get("transformations")
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        images = inputs.get(IMAGES, None)
        raw_images = inputs.get("raw_images", None)
        segmentation_masks = segmentation_masks.to_tensor()
        segmentation_masks = self.augment_ragged_segmentation_mask(
            segmentation_mask=segmentation_masks,
            transformation=transformations,
            label=labels,
            bounding_boxes=bounding_boxes,
            image=images,
            raw_image=raw_images,
        )
        return tf.RaggedTensor.from_tensor(segmentation_masks)

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
        raw_images = images
        labels = inputs.get(LABELS, None)
        bounding_boxes = inputs.get(BOUNDING_BOXES, None)
        keypoints = inputs.get(KEYPOINTS, None)
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        custom_annotations = inputs.get(CUSTOM_ANNOTATIONS, None)
        batch_size = tf.shape(images)[0]

        transformations = self.get_random_transformation_batch(
            batch_size,
            images=images,
            labels=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_masks=segmentation_masks,
            custom_annotations=custom_annotations,
        )

        if (
            isinstance(images, tf.RaggedTensor)
            and not self.force_no_unwrap_ragged_image_call
        ):
            inputs_for_raggeds = {"transformations": transformations, **inputs}
            images = tf.map_fn(
                self._unwrap_ragged_image_call,
                inputs_for_raggeds,
                fn_output_signature=self.compute_ragged_image_signature(images),
            )
        else:
            images = self.augment_images(
                images,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                labels=labels,
            )
        if (
            isinstance(images, tf.RaggedTensor)
            and self.force_output_dense_images
        ):
            images = images.to_tensor()
        result = {IMAGES: images}

        if labels is not None:
            labels = self.augment_labels(
                labels,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[LABELS] = labels

        if bounding_boxes is not None:
            ori_bounding_boxes_info = bounding_box.validate_format(
                bounding_boxes
            )
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=images,
                raw_images=raw_images,
            )
            if ori_bounding_boxes_info["ragged"]:
                bounding_boxes = bounding_box.to_ragged(
                    bounding_boxes, dtype=self.compute_dtype
                )
            result[BOUNDING_BOXES] = bounding_boxes

        if keypoints is not None:
            keypoints = self.augment_keypoints(
                keypoints,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[KEYPOINTS] = keypoints

        if segmentation_masks is not None:
            if (
                isinstance(segmentation_masks, tf.RaggedTensor)
                and not self.force_no_unwrap_ragged_image_call
            ):
                inputs_for_raggeds = {
                    "transformations": transformations,
                    "raw_images": raw_images,
                    **inputs,
                }
                segmentation_masks = tf.map_fn(
                    self._unwrap_ragged_segmentation_mask_call,
                    inputs_for_raggeds,
                    fn_output_signature=self.compute_ragged_segmentation_mask_signature(  # noqa: E501
                        segmentation_masks
                    ),
                )
            else:
                segmentation_masks = self.augment_segmentation_masks(
                    segmentation_masks,
                    transformations=transformations,
                    labels=labels,
                    bounding_boxes=bounding_boxes,
                    images=images,
                    raw_images=raw_images,
                )
            if (
                isinstance(segmentation_masks, tf.RaggedTensor)
                and self.force_output_dense_images
            ):
                segmentation_masks = segmentation_masks.to_tensor()
            result[SEGMENTATION_MASKS] = segmentation_masks

        if custom_annotations is not None:
            custom_annotations = self.augment_custom_annotations(
                custom_annotations,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[CUSTOM_ANNOTATIONS] = custom_annotations

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def _ensure_inputs_are_compute_dtype(self, inputs):
        if not isinstance(inputs, dict):
            return preprocessing.ensure_tensor(
                inputs,
                self.compute_dtype,
            )
        inputs[IMAGES] = preprocessing.ensure_tensor(
            inputs[IMAGES],
            self.compute_dtype,
        )
        if BOUNDING_BOXES in inputs:
            inputs[BOUNDING_BOXES]["boxes"] = preprocessing.ensure_tensor(
                inputs[BOUNDING_BOXES]["boxes"],
                self.compute_dtype,
            )
            inputs[BOUNDING_BOXES]["classes"] = preprocessing.ensure_tensor(
                inputs[BOUNDING_BOXES]["classes"],
                self.compute_dtype,
            )
        return inputs

    def _format_inputs(self, inputs):
        metadata = {IS_DICT: True, USE_TARGETS: False}
        if tf.is_tensor(inputs):
            metadata[IS_DICT] = False
            inputs = {IMAGES: inputs}

        metadata[BATCHED] = inputs["images"].shape.rank == 4
        if inputs["images"].shape.rank == 3:
            for key in list(inputs.keys()):
                if key == BOUNDING_BOXES:
                    inputs[BOUNDING_BOXES]["boxes"] = tf.expand_dims(
                        inputs[BOUNDING_BOXES]["boxes"], axis=0
                    )
                    inputs[BOUNDING_BOXES]["classes"] = tf.expand_dims(
                        inputs[BOUNDING_BOXES]["classes"], axis=0
                    )
                else:
                    inputs[key] = tf.expand_dims(inputs[key], axis=0)

        if not isinstance(inputs, dict):
            raise ValueError(
                "Expect the inputs to be image tensor or dict. Got "
                f"inputs={inputs}"
            )

        if BOUNDING_BOXES in inputs:
            inputs[BOUNDING_BOXES] = self._format_bounding_boxes(
                inputs[BOUNDING_BOXES]
            )

        if isinstance(inputs, dict) and TARGETS in inputs:
            inputs[LABELS] = inputs[TARGETS]
            del inputs[TARGETS]
            metadata[USE_TARGETS] = True
            return inputs, metadata

        return inputs, metadata

    def _format_bounding_boxes(self, bounding_boxes):
        # We can't catch the case where this is None, sometimes RaggedTensor
        # drops this dimension.
        if "classes" not in bounding_boxes:
            raise ValueError(
                "Bounding boxes are missing class_id. If you would like to pad "
                "the bounding boxes with class_id, use: "
                "`bounding_boxes['classes'] = "
                "tf.ones_like(bounding_boxes['boxes'])`."
            )
        return bounding_boxes

    def _format_output(self, output, metadata):
        if not metadata[BATCHED]:
            for key in list(output.keys()):
                if key == BOUNDING_BOXES:
                    output[BOUNDING_BOXES]["boxes"] = tf.squeeze(
                        output[BOUNDING_BOXES]["boxes"], axis=0
                    )
                    output[BOUNDING_BOXES]["classes"] = tf.squeeze(
                        output[BOUNDING_BOXES]["classes"], axis=0
                    )
                else:
                    output[key] = tf.squeeze(output[key], axis=0)

        if not metadata[IS_DICT]:
            return output[IMAGES]
        elif metadata[USE_TARGETS]:
            output[TARGETS] = output[LABELS]
            del output[LABELS]
        return output

    def call(self, inputs):
        inputs = self._ensure_inputs_are_compute_dtype(inputs)
        inputs, metadata = self._format_inputs(inputs)
        images = inputs[IMAGES]
        if images.shape.rank == 3 or images.shape.rank == 4:
            return self._format_output(self._batch_augment(inputs), metadata)
        else:
            raise ValueError(
                "Image augmentation layers are expecting inputs to be "
                "rank 3 (HWC) or 4D (NHWC) tensors. Got shape: "
                f"{images.shape}"
            )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

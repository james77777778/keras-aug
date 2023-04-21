import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing
from tensorflow import keras

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
class VectorizedBaseRandomLayer(keras.__internal__.layers.BaseRandomLayer):
    """Abstract base layer for vectorized image augmentation.

    This layer contains base functionalities for preprocessing layers which
    augment image related data, e.g. image and in the future, label and bounding
    boxes. The subclasses could avoid making certain mistakes and reduce code
    duplications.

    This layer requires you to implement one method `augment_images()`, which
    augments one single image during the training. There are a few additional
    methods that you can implement for added functionality on the layer.

    `augment_ragged_image()` and `compute_ragged_image_signature()`, which
    handles ragged images augmentation if the layer supports that.

    `augment_labels()`, which handles label augmentation if the layer
    supports that.

    `augment_bounding_boxes()`, which handles the bounding box
    augmentation, if the layer supports that.

    `augment_keypoints()`, which handles the keypoints augmentation, if the
    layer supports that.

    `augment_segmentation_masks()`, which handles the segmentation masks
    augmentation, if the layer supports that.

    `augment_custom_annotations()`, which handles the custom annotations
    augmentation, if the layer supports that. This is useful to implement
    augmentation for special annotatinos.

    `get_random_transformations()`, which should produce a batch of random
    transformation settings. The transformation object, which must be a
    batched Tensor or a dictionary where each input is a batched Tensor,
    will be passed to `augment_images`, `augment_labels` and
    `augment_bounding_boxes`, to coordinate the randomness behavior, eg, in
    the RandomFlip layer, the image and bounding_boxes should be changed in
    the same way.

    The `call()` method support two formats of inputs:
        1. Single image tensor with 3D (HWC) or 4D (NHWC) format.
        2. A dict of tensors with stable keys. The supported keys are:
            `"images"`, `"labels"`, `"bounding_boxes"`, `segmentation_masks`,
            `keypoints` and `custom_annotations` at the moment. We might add
            more keys in future when we support more types of augmentation.

    The output of the `call()` will be in two formats, which will be the same
    structure as the inputs.

    The `call()` will unpack the inputs, forward to the correct function, and
    pack the output back to the same structure as the inputs.

    Note that since the randomness is also a common functionality, this layer
    also includes a keras.backend.RandomGenerator, which can be used to
    produce the random numbers. The random number generator is stored in the
    `self._random_generator` attribute.
    """

    def __init__(
        self,
        force_no_unwrap_ragged_image_call=False,
        force_output_dense_images=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.force_no_unwrap_ragged_image_call = (
            force_no_unwrap_ragged_image_call
        )
        self.force_output_dense_images = force_output_dense_images

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
        """Computes the output image signature for the `augment_image()`
        function.

        Must be overridden to return tensors with different shapes than the
        input images. By default, returns either a `tf.RaggedTensorSpec`
        matching the input image spec, or a `tf.TensorSpec` matching the input
        image spec.
        """
        ragged_spec = tf.RaggedTensorSpec(
            shape=images.shape[1:],
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

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
        transformation = inputs.get("transformations")
        images = images.to_tensor()
        images = self.augment_ragged_image(
            image=images,
            label=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_mask=segmentation_masks,
            transformation=transformation,
        )
        return tf.RaggedTensor.from_tensor(images)

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
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=images,
                raw_images=raw_images,
            )
            bounding_boxes = bounding_box.to_ragged(bounding_boxes)
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
            segmentation_masks = self.augment_segmentation_masks(
                segmentation_masks,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
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
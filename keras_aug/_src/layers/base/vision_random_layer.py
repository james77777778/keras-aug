import keras
from keras import backend
from keras import tree
from keras.src.utils.backend_utils import in_tf_graph

from keras_aug._src.backend.bounding_box import BoundingBoxBackend
from keras_aug._src.backend.dynamic_backend import DynamicBackend
from keras_aug._src.backend.dynamic_backend import DynamicRandomGenerator
from keras_aug._src.backend.image import ImageBackend
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.layers.base", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class VisionRandomLayer(keras.Layer):
    """Abstract base layer for vectorized image augmentation.

    This layer contains base functionalities for preprocessing layers which
    augment image related data, e.g. image and in the future, label and bounding
    boxes. The subclasses could avoid making certain mistakes and reduce code
    duplications.

    This layer requires you to implement one method `augment_images()`, which
    augments one single image during the training. There are a few additional
    methods that you can implement for added functionality on the layer.

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

    The `call()` method support two formats of inputs::

        1. Single image tensor with 3D or 4D format.
        2. A dict of tensors with stable keys. The supported keys are
            `"images"`, `"labels"`, `"bounding_boxes"`,
            `segmentation_masks`, `keypoints` and `custom_annotations` at
            the moment. We might add more keys in future when we support more
            types of augmentation.

    The output of the `call()` will be in two formats, which will be the same
    structure as the inputs.

    The `call()` will unpack the inputs, forward to the correct function, and
    pack the output back to the same structure as the inputs.
    """

    IMAGES = "images"
    LABELS = "labels"
    BOUNDING_BOXES = "bounding_boxes"
    KEYPOINTS = "keypoints"
    SEGMENTATION_MASKS = "segmentation_masks"
    CUSTOM_ANNOTATIONS = "custom_annotations"

    IS_DICT = "is_dict"
    BATCHED = "batched"

    def __init__(self, has_generator=True, seed=None, **kwargs):
        super().__init__(**kwargs)
        self._backend = DynamicBackend(backend.backend())
        if has_generator:
            self._random_generator = DynamicRandomGenerator(
                backend.backend(), seed=seed
            )
        self.has_generator = has_generator
        self.seed = seed

        # Other vision backends
        self.image_backend = ImageBackend(backend.backend())
        self.bbox_backend = BoundingBoxBackend(backend.backend())

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.autocast = False

    @property
    def image_dtype(self):
        return self.compute_dtype

    @property
    def bounding_box_dtype(self):
        dtype = backend.result_type(self.compute_dtype, float)
        return dtype

    @property
    def keypoint_dtype(self):
        dtype = backend.result_type(self.compute_dtype, float)
        return dtype

    @property
    def backend(self):
        return self._backend.backend

    @property
    def random_generator(self):
        return self._random_generator.random_generator

    def get_params(
        self,
        batch_size,
        images=None,
        labels=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_masks=None,
        custom_annotations=None,
    ):
        """Produce transformations parameters.

        Returns:
            Any type of object, which will be forwarded as the
            `transformations` parameter.
        """
        return self.backend.numpy.zeros([batch_size])

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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support augmenting images."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support augmenting labels."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support augmenting "
            "bounding boxes."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support augmenting "
            "keypoints."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support augmenting "
            "segmentation masks."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support augmenting "
            "custom annotations."
        )

    def _batch_augment(self, inputs, **kwargs):
        images = inputs.get(self.IMAGES, None)
        raw_images = images
        labels = inputs.get(self.LABELS, None)
        bounding_boxes = inputs.get(self.BOUNDING_BOXES, None)
        keypoints = inputs.get(self.KEYPOINTS, None)
        segmentation_masks = inputs.get(self.SEGMENTATION_MASKS, None)
        custom_annotations = inputs.get(self.CUSTOM_ANNOTATIONS, None)
        batch_size = self.backend.shape(images)[0]

        transformations = self.get_params(
            batch_size,
            images=images,
            labels=labels,
            bounding_boxes=bounding_boxes,
            keypoints=keypoints,
            segmentation_masks=segmentation_masks,
            custom_annotations=custom_annotations,
        )

        images = self.augment_images(
            images,
            transformations=transformations,
            bounding_boxes=bounding_boxes,
            labels=labels,
        )
        result = {self.IMAGES: images}

        if labels is not None:
            labels = self.augment_labels(
                labels,
                transformations=transformations,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[self.LABELS] = labels

        if bounding_boxes is not None:
            bounding_boxes = self.augment_bounding_boxes(
                bounding_boxes,
                transformations=transformations,
                labels=labels,
                images=images,
                raw_images=raw_images,
            )
            result[self.BOUNDING_BOXES] = bounding_boxes

        if keypoints is not None:
            keypoints = self.augment_keypoints(
                keypoints,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[self.KEYPOINTS] = keypoints

        if segmentation_masks is not None:
            segmentation_masks = self.augment_segmentation_masks(
                segmentation_masks,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[self.SEGMENTATION_MASKS] = segmentation_masks

        if custom_annotations is not None:
            custom_annotations = self.augment_custom_annotations(
                custom_annotations,
                transformations=transformations,
                labels=labels,
                bounding_boxes=bounding_boxes,
                images=images,
                raw_images=raw_images,
            )
            result[self.CUSTOM_ANNOTATIONS] = custom_annotations

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def _format_inputs(self, inputs):
        ops = self.backend
        metadata = {self.IS_DICT: True}
        if not isinstance(inputs, dict):
            metadata[self.IS_DICT] = False
            inputs = {self.IMAGES: inputs}
        if not isinstance(inputs, dict):
            raise ValueError(
                "Expect the inputs to be image tensor or dict. "
                f"Received: inputs={inputs} of type={type(inputs)}"
            )

        images_shape = ops.shape(inputs["images"])
        metadata[self.BATCHED] = len(images_shape) == 4
        if len(images_shape) == 3:
            for key in list(inputs.keys()):
                if key == self.BOUNDING_BOXES:
                    inputs[self.BOUNDING_BOXES]["boxes"] = (
                        ops.numpy.expand_dims(
                            inputs[self.BOUNDING_BOXES]["boxes"], axis=0
                        )
                    )
                    inputs[self.BOUNDING_BOXES]["classes"] = (
                        ops.numpy.expand_dims(
                            inputs[self.BOUNDING_BOXES]["classes"], axis=0
                        )
                    )
                else:
                    inputs[key] = ops.numpy.expand_dims(inputs[key], axis=0)
        if self.BOUNDING_BOXES in inputs:
            inputs[self.BOUNDING_BOXES] = self._format_bounding_boxes(
                inputs[self.BOUNDING_BOXES]
            )
        return inputs, metadata

    def _cast_inputs(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError
        ops = self.backend
        inputs = inputs.copy()
        if self.IMAGES in inputs:
            inputs[self.IMAGES] = ops.convert_to_tensor(inputs[self.IMAGES])
            inputs[self.IMAGES] = self.image_backend.transform_dtype(
                inputs[self.IMAGES], self.image_dtype
            )
        if self.LABELS in inputs:
            inputs[self.LABELS] = ops.convert_to_tensor(inputs[self.LABELS])
        if self.BOUNDING_BOXES in inputs:
            bounding_boxes = inputs[self.BOUNDING_BOXES].copy()
            bounding_boxes["boxes"] = ops.convert_to_tensor(
                bounding_boxes["boxes"], self.bounding_box_dtype
            )
            bounding_boxes["classes"] = ops.convert_to_tensor(
                bounding_boxes["classes"], self.bounding_box_dtype
            )
            inputs[self.BOUNDING_BOXES] = bounding_boxes
        if self.SEGMENTATION_MASKS in inputs:
            masks = inputs[self.SEGMENTATION_MASKS]
            masks = ops.convert_to_tensor(masks)
            if backend.is_float_dtype(masks.dtype) and backend.is_float_dtype(
                self.compute_dtype
            ):
                masks = ops.cast(masks, self.compute_dtype)
            inputs[self.SEGMENTATION_MASKS] = masks
        if self.KEYPOINTS in inputs:
            inputs[self.KEYPOINTS] = ops.convert_to_tensor(
                inputs[self.KEYPOINTS], self.keypoint_dtype
            )
        if self.CUSTOM_ANNOTATIONS in inputs:
            inputs[self.CUSTOM_ANNOTATIONS] = tree.map_structure(
                ops.convert_to_tensor, inputs[self.CUSTOM_ANNOTATIONS]
            )
        return inputs

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
        ops = self.backend
        if not metadata[self.BATCHED]:
            for key in list(output.keys()):
                if key == self.BOUNDING_BOXES:
                    output[self.BOUNDING_BOXES]["boxes"] = ops.numpy.squeeze(
                        output[self.BOUNDING_BOXES]["boxes"], axis=0
                    )
                    output[self.BOUNDING_BOXES]["classes"] = ops.numpy.squeeze(
                        output[self.BOUNDING_BOXES]["classes"], axis=0
                    )
                else:
                    output[key] = ops.numpy.squeeze(output[key], axis=0)
        if not metadata[self.IS_DICT]:
            return output[self.IMAGES]
        return output

    def __call__(self, inputs, **kwargs):
        if in_tf_graph():
            self._set_backend("tensorflow")
            try:
                outputs = super().__call__(inputs, **kwargs)
            finally:
                self._reset_backend()
            return outputs
        else:
            return super().__call__(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        ops = self.backend
        inputs, metadata = self._format_inputs(inputs)
        inputs = self._cast_inputs(inputs)
        images = inputs[self.IMAGES]
        images_shape = ops.shape(images)
        if len(images_shape) == 4:
            return self._format_output(self._batch_augment(inputs), metadata)
        else:
            raise ValueError(
                "Image augmentation layers are expecting inputs to be "
                "rank 3D (unbatched) or 4D (batched) tensors. "
                f"Received: images.shape={images.shape}"
            )

    def get_config(self):
        config = super().get_config()
        config.update({"seed": self.seed})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # Useful functions

    def transform_value_range(
        self, images, original_range, target_range, dtype="float32"
    ):
        ops = self.backend
        original_range = tuple(original_range)
        target_range = tuple(target_range)

        images = ops.cast(images, dtype)
        images = ops.numpy.divide(
            (images - original_range[0]),
            (original_range[1] - original_range[0]),
        )
        scale_factor = ops.numpy.subtract(target_range[1], target_range[0])
        images = ops.numpy.add(
            ops.numpy.multiply(images, scale_factor), target_range[0]
        )
        return images

    def _set_backend(self, name):
        self._backend.set_backend(name)
        self.image_backend.set_backend(name)
        self.bbox_backend.set_backend(name)
        if self.has_generator:
            self._random_generator.set_generator(name)

    def _reset_backend(self):
        self._backend.reset()
        self.image_backend.reset()
        self.bbox_backend.reset()
        if self.has_generator:
            self._random_generator.reset()

    def _get_shape_or_spec(self, input_shape_or_inputs):
        """Get the shape or spec of `images` and `segmentation_masks`."""
        if not isinstance(input_shape_or_inputs, dict):
            return input_shape_or_inputs, None
        else:
            images = input_shape_or_inputs[self.IMAGES]
            segmentation_masks = input_shape_or_inputs.get(
                self.SEGMENTATION_MASKS, None
            )
            return images, segmentation_masks

    def _set_shape(
        self, original_shape, images_shape, segmentation_masks_shape=None
    ):
        images_shape = tuple(images_shape)
        if segmentation_masks_shape is not None:
            segmentation_masks_shape = tuple(segmentation_masks_shape)
        if not isinstance(original_shape, dict):
            return images_shape
        else:
            original_shape[self.IMAGES] = images_shape
            if self.SEGMENTATION_MASKS in original_shape:
                original_shape[self.SEGMENTATION_MASKS] = (
                    segmentation_masks_shape
                )
            return original_shape

    def _set_spec(
        self, original_spec, images_spec, segmentation_masks_spec=None
    ):
        if not isinstance(original_spec, dict):
            return images_spec
        else:
            original_spec[self.IMAGES] = images_spec
            if self.SEGMENTATION_MASKS in original_spec:
                original_spec[self.SEGMENTATION_MASKS] = segmentation_masks_spec
            return original_spec

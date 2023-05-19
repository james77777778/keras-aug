import keras_cv
import tensorflow as tf

try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None

from keras_aug.utils.conditional_imports import assert_tfds_installed


def load_voc_dataset(
    bounding_box_format,
    name="voc/2007",
    batch_size=9,
):
    assert_tfds_installed("load_voc_dataset")

    def preprocess_voc(inputs, format=None):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        if format is not None:
            boxes = inputs["objects"]["bbox"]
            boxes = keras_cv.bounding_box.convert_format(
                boxes,
                images=image,
                source="rel_yxyx",
                target=format,
            )
            classes = tf.cast(inputs["objects"]["label"], tf.float32)
            bounding_boxes = {"classes": classes, "boxes": boxes}
            return {"images": image, "bounding_boxes": bounding_boxes}
        else:
            return {"images": image}

    dataset = tfds.load(name, split=tfds.Split.TRAIN, shuffle_files=False)
    dataset = dataset.map(
        lambda x: preprocess_voc(x, format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if tf.__version__ >= "2.12.0":
        dataset = dataset.ragged_batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(
                batch_size, drop_remainder=True
            )
        )
    return dataset


def load_oxford_dataset(
    name="oxford_flowers102",
    batch_size=9,
    img_size=(224, 224),
    as_supervised=True,
):
    assert_tfds_installed("oxford_dataset")

    def preprocess_oxford(image, label, img_size=(224, 224), num_classes=10):
        image = tf.image.resize(image, img_size)
        label = tf.one_hot(label, num_classes)
        return {"images": image, "labels": label}

    data, ds_info = tfds.load(
        name, as_supervised=as_supervised, with_info=True, shuffle_files=False
    )
    dataset = data["train"]
    num_classes = ds_info.features["label"].num_classes
    dataset = dataset.map(
        lambda x, y: preprocess_oxford(
            x, y, img_size=img_size, num_classes=num_classes
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    return dataset


def load_oxford_iiit_pet_dataset(
    name="oxford_iiit_pet:3.*.*",
    batch_size=9,
    img_size=(224, 224),
):
    assert_tfds_installed("oxford_iiit_pet_dataset")

    def preprocess_oxford_iiit_pet(data):
        img_size = (224, 224)

        input_image = tf.image.resize(data["image"], img_size)
        input_mask = tf.image.resize(
            data["segmentation_mask"],
            img_size,
            method="nearest",
        )
        input_image = tf.image.convert_image_dtype(input_image, tf.float32)
        input_mask -= 1
        input_mask = tf.cast(input_mask, tf.float32)
        return {"images": input_image, "segmentation_masks": input_mask}

    data = tfds.load(name)
    dataset = data["train"]
    dataset = dataset.map(
        preprocess_oxford_iiit_pet,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    return dataset


def visualize_data(
    data, value_range=(0, 255), bounding_box_format=None, output_path=None
):
    data = next(iter(data))
    images = data["images"]
    if isinstance(images, tf.RaggedTensor):
        images = images.to_tensor(0)
    if "labels" in data:
        mask = tf.greater(data["labels"][0], 0)
        non_zero_array = tf.boolean_mask(data["labels"][0], mask)
        print(f"Nonzero labels of the first image:\n{non_zero_array}")
    if bounding_box_format is not None:
        bounding_boxes = data["bounding_boxes"]
        keras_cv.visualization.plot_bounding_box_gallery(
            images,
            value_range=value_range,
            bounding_box_format=bounding_box_format,
            y_true=bounding_boxes,
            path=output_path,
            dpi=100,
        )
    else:
        keras_cv.visualization.plot_image_gallery(
            images,
            value_range=value_range,
            path=output_path,
            dpi=100,
        )


def visualize_data_single(
    data, value_range=(0, 255), bounding_box_format=None, output_path=None
):
    data = next(iter(data))
    images = data["images"]
    images = images[0:1, ...]
    if isinstance(images, tf.RaggedTensor):
        images = images.to_tensor(0)
    if "labels" in data:
        mask = tf.greater(data["labels"][0], 0)
        non_zero_array = tf.boolean_mask(data["labels"][0], mask)
        print(f"Nonzero labels of the first image:\n{non_zero_array}")
    if bounding_box_format is not None:
        bounding_boxes = data["bounding_boxes"]
        bounding_boxes = {
            "boxes": bounding_boxes["boxes"][0:1, ...],
            "classes": bounding_boxes["classes"][0:1, ...],
        }
        keras_cv.visualization.plot_bounding_box_gallery(
            images,
            value_range=value_range,
            bounding_box_format=bounding_box_format,
            y_true=bounding_boxes,
            path=output_path,
            line_thickness=1,
            rows=1,
            cols=1,
            dpi=300,
        )
    else:
        keras_cv.visualization.plot_image_gallery(
            images,
            value_range=value_range,
            path=output_path,
            rows=1,
            cols=1,
            dpi=300,
        )


def visualize_segmentation_masks(
    data,
    image_value_range=(0, 255),
    mask_value_range=None,
    output_path=None,
):
    data = next(iter(data))
    images = data["images"]
    masks = data["segmentation_masks"]
    masks = keras_cv.utils.transform_value_range(
        masks, mask_value_range, (0, 255)
    )
    masks = tf.concat([masks, masks, masks], axis=-1)
    display_images = tf.concat([images, masks], axis=2)  # B, H, W, C

    keras_cv.visualization.plot_image_gallery(
        display_images,
        value_range=image_value_range,
        path=output_path,
        dpi=100,
    )

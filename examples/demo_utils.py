"""
References:
 - https://github.com/keras-team/keras-cv/blob/master/examples/layers/preprocessing/bounding_box/demo_utils.py
"""  # noqa: E501

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv import bounding_box


def preprocess_voc(inputs, format=None):
    image = inputs["image"]
    image = tf.cast(image, tf.float32)
    if format is not None:
        boxes = inputs["objects"]["bbox"]
        boxes = bounding_box.convert_format(
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


def load_voc_dataset(
    bounding_box_format,
    name="voc/2007",
    batch_size=9,
):
    dataset = tfds.load(name, split=tfds.Split.TRAIN, shuffle_files=True)
    dataset = dataset.map(
        lambda x: preprocess_voc(x, format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(
            batch_size, drop_remainder=True
        )
    )
    return dataset


def visualize_data(data, bounding_box_format, output_path=None):
    data = next(iter(data))
    images = data["images"]
    if bounding_box_format is not None:
        bounding_boxes = data["bounding_boxes"]
        output_images = visualize_bounding_boxes(
            images, bounding_boxes, bounding_box_format
        ).numpy()
    else:
        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(0)
        output_images = images.numpy()
    gallery_show(output_images.astype(int), output_path)


def visualize_data_across_batch(data, bounding_box_format, output_path=None):
    results = []
    data_iterator = iter(data)
    for _ in range(9):
        cur_data = next(data_iterator)
        images = cur_data["images"]
        if bounding_box_format is not None:
            bounding_boxes = cur_data["bounding_boxes"]
            output_images = visualize_bounding_boxes(
                images, bounding_boxes, bounding_box_format
            ).numpy()
        else:
            if isinstance(images, tf.RaggedTensor):
                images = images.to_tensor(0)
            output_images = images.numpy()
        # pick first output_image
        results.append(output_images[0].astype(int))
    gallery_show(results, output_path)


def visualize_bounding_boxes(image, bounding_boxes, bounding_box_format):
    color = np.array([[255.0, 0.0, 0.0]])
    bounding_boxes = bounding_box.to_dense(bounding_boxes)
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="yxyx",
        images=image,
    )
    if isinstance(image, tf.RaggedTensor):
        image = image.to_tensor(0)
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source="yxyx",
        target="rel_yxyx",
        images=image,
    )
    bounding_boxes = bounding_boxes["boxes"]
    return tf.image.draw_bounding_boxes(image, bounding_boxes, color, name=None)


def gallery_show(images, output_path=None):
    for i in range(9):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
    plt.tight_layout()
    if output_path is None:
        output_path = "demo.png"
    plt.savefig(output_path, dpi=150)

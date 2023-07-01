import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import keras_aug

BATCH_SIZE = 16
OUTPUT_PATH = "output.png"


def visualize_dataset(
    inputs, value_range, rows, cols, bounding_box_format, path
):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        path=path,
        dpi=150,
    )


def unpackage_raw_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_aug.datapoints.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {
        "images": tf.cast(image, tf.float32),
        "bounding_boxes": bounding_boxes,
    }


def load_pascal_voc(split, dataset, bounding_box_format):
    ds = tfds.load(dataset, split=split, with_info=False, shuffle_files=False)
    ds = ds.map(
        lambda x: unpackage_raw_tfds_inputs(
            x, bounding_box_format=bounding_box_format
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


augmenter = keras.Sequential(
    layers=[
        keras_aug.layers.Resize(
            640,
            640,
            pad_to_aspect_ratio=True,
            padding_value=114,
            bounding_box_format="xywh",
        ),
        keras_aug.layers.RandomFlip(bounding_box_format="xywh"),
        keras_aug.layers.RandomApply(
            layer=keras_aug.layers.RandomColorJitter(
                value_range=(0, 255), brightness_factor=(1.5, 1.5)
            ),
            rate=0.9,
        ),
        keras_aug.layers.RandomChoice(
            layers=[
                keras_aug.layers.ChannelShuffle(),
                keras_aug.layers.RandomChannelShift(
                    value_range=(0, 255), factor=0.2
                ),
            ]
        ),
        keras_aug.layers.RandomChoice(
            layers=[
                keras_aug.layers.Mosaic(
                    height=640,
                    width=640,
                    fill_value=114,
                    bounding_box_format="xywh",
                ),
                keras_aug.layers.MixUp(alpha=32.0),
            ],
            batchwise=True,
        ),
    ]
)


train_ds = load_pascal_voc(
    split="train", dataset="voc/2007", bounding_box_format="xywh"
)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

for i in range(3):
    visualize_dataset(
        train_ds,
        bounding_box_format="xywh",
        value_range=(0, 255),
        rows=2,
        cols=2,
        path=f"{i}_{OUTPUT_PATH}",
    )

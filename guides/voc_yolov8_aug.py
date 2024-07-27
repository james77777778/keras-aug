import cv2
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_aug import layers as ka_layers
from keras_aug import ops as ka_ops
from keras_aug import visualization


def load_voc(name, split, shuffle, batch_size, position):
    def unpack_voc_inputs(x):
        image = x["image"]
        image_shape = tf.shape(image)
        height, width = image_shape[-3], image_shape[-2]
        boxes = ka_ops.bounding_box.convert_format(
            x["objects"]["bbox"],
            source="rel_yxyx",
            target="xyxy",
            height=height,
            width=width,
        )
        bounding_boxes = {"classes": x["objects"]["label"], "boxes": boxes}
        return {"images": image, "bounding_boxes": bounding_boxes}

    ds = tfds.load(name, split=split, with_info=False, shuffle_files=shuffle)
    ds: tf.data.Dataset = ds.map(lambda x: unpack_voc_inputs(x))

    # You can utilize KerasAug's layers in `tf.data` pipeline.
    # The layer will automatically switch to the TensorFlow backend to be
    # compatible with `tf.data`.
    ds = ds.map(ka_layers.vision.MaxBoundingBox(40))  # Max: 37 in train
    ds = ds.shuffle(128, reshuffle_each_iteration=True)
    ds = ds.map(
        ka_layers.vision.Resize(
            640, along_long_edge=True, bounding_box_format="xyxy", dtype="uint8"
        )
    )
    ds = ds.map(
        ka_layers.vision.Pad(
            (640, 640),
            padding_position=position,
            padding_value=114,
            bounding_box_format="xyxy",
            dtype="uint8",
        )
    )
    ds = ds.batch(batch_size)
    return ds


args = dict(name="voc/2007", split="train", shuffle=True, batch_size=16)
ds_tl = load_voc(**args, position="top_left")
ds_tr = load_voc(**args, position="top_right")
ds_bl = load_voc(**args, position="bottom_left")
ds_br = load_voc(**args, position="bottom_right")
ds = tf.data.Dataset.zip(ds_tl, ds_tr, ds_bl, ds_br)
ds = ds.map(
    ka_layers.vision.Mosaic(
        (1280, 1280),
        offset=(0.25, 0.75),
        padding_value=114,
        bounding_box_format="xyxy",
        dtype="uint8",
    )
)

# You can also utilize KerasAug's layers in a typical Keras manner.
# `augmenter`` will be called just like a regular Keras model, benefiting from
# accelerator (such as GPU & TPU) and compilation.
augmenter = keras.Sequential(
    [
        ka_layers.vision.RandomAffine(
            translate=0.05,
            scale=0.25,
            padding_value=114,
            bounding_box_format="xyxy",
            dtype="uint8",
        ),
        ka_layers.vision.CenterCrop(
            (640, 640), bounding_box_format="xyxy", dtype="uint8"
        ),
        ka_layers.vision.RandomGrayscale(p=0.01),
        ka_layers.vision.RandomHSV(hue=0.015, saturation=0.7, value=0.4),
        ka_layers.vision.RandomFlip(
            mode="horizontal", bounding_box_format="xyxy"
        ),
    ]
)

for x in ds.take(1):
    x = augmenter(x)
    drawed_images = visualization.draw_bounding_boxes(
        x["images"], x["bounding_boxes"], bounding_box_format="xyxy"
    )
    cv2.imwrite("output.jpg", drawed_images[0])
    for i_d in range(drawed_images.shape[0]):
        output_path = f"output_{i_d}.jpg"
        output_image = cv2.cvtColor(drawed_images[i_d], cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)

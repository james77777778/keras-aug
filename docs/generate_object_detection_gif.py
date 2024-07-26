import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from keras_aug import layers as ka_layers
from keras_aug import ops as ka_ops
from keras_aug import visualization

size = (320, 320)
mosaic_size = (640, 640)


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
    ds = ds.map(ka_layers.vision.MaxBoundingBox(40))  # Max: 37 in train
    ds = ds.shuffle(128, reshuffle_each_iteration=True)
    ds = ds.map(
        ka_layers.vision.Resize(
            size[0],
            along_long_edge=True,
            bounding_box_format="xyxy",
            dtype="uint8",
        )
    )
    ds = ds.map(
        ka_layers.vision.Pad(
            size,
            padding_position=position,
            padding_value=114,
            bounding_box_format="xyxy",
            dtype="uint8",
        )
    )
    ds = ds.batch(batch_size)
    return ds


# Load dataset
args = dict(name="voc/2007", split="train", shuffle=True, batch_size=16)
ds_tl = load_voc(**args, position="top_left")
ds_tr = load_voc(**args, position="top_right")
ds_bl = load_voc(**args, position="bottom_left")
ds_br = load_voc(**args, position="bottom_right")
ds = tf.data.Dataset.zip(ds_tl, ds_tr, ds_bl, ds_br)

# Augment
ds = ds.map(
    ka_layers.vision.Mosaic(
        mosaic_size,
        offset=(0.25, 0.75),
        padding_value=114,
        bounding_box_format="xyxy",
        dtype="uint8",
    )
)
ds = ds.map(
    ka_layers.vision.RandomAffine(
        translate=0.05,
        scale=0.25,
        padding_value=114,
        bounding_box_format="xyxy",
        dtype="uint8",
    )
)
ds = ds.map(
    ka_layers.vision.CenterCrop(size, bounding_box_format="xyxy", dtype="uint8")
)
ds = ds.map(ka_layers.vision.RandomGrayscale(p=0.01))
ds = ds.map(ka_layers.vision.RandomHSV(hue=0.015, saturation=0.7, value=0.4))
ds = ds.map(
    ka_layers.vision.RandomFlip(mode="horizontal", bounding_box_format="xyxy")
)

# Make gif
images = []
for x in ds.take(1):
    drawed_images = visualization.draw_bounding_boxes(
        x["images"], x["bounding_boxes"], bounding_box_format="xyxy"
    )
    for i in range(drawed_images.shape[0]):
        images.append(Image.fromarray(drawed_images[i]))
images[0].save(
    "output.gif",
    save_all=True,
    append_images=images[1:10],
    optimize=False,
    duration=1000,
    loop=0,
)

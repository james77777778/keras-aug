import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from keras_aug import layers as ka_layers
from keras_aug import visualization

size = (320, 320)
mosaic_size = (640, 640)


def load_oxford(name, split, shuffle, batch_size, position):
    def unpack_oxford_inputs(x):
        segmentation_masks = tf.cast(x["segmentation_mask"], "int8")
        segmentation_masks = tf.where(
            tf.equal(segmentation_masks, 2),  # Background index
            tf.constant(-1, dtype=segmentation_masks.dtype),
            segmentation_masks,
        )
        return {
            "images": x["image"],
            "segmentation_masks": segmentation_masks,
        }

    ds = tfds.load(name, split=split, with_info=False, shuffle_files=shuffle)
    ds: tf.data.Dataset = ds.map(lambda x: unpack_oxford_inputs(x))
    ds = ds.shuffle(128, reshuffle_each_iteration=True)
    ds = ds.map(
        ka_layers.vision.Resize(size[0], along_long_edge=True, dtype="uint8")
    )
    ds = ds.map(
        ka_layers.vision.Pad(
            size, padding_position=position, padding_value=114, dtype="uint8"
        )
    )
    ds = ds.batch(batch_size)
    return ds


args = dict(name="oxford_iiit_pet", split="train", shuffle=True, batch_size=16)
ds_tl = load_oxford(**args, position="top_left")
ds_tr = load_oxford(**args, position="top_right")
ds_bl = load_oxford(**args, position="bottom_left")
ds_br = load_oxford(**args, position="bottom_right")
ds = tf.data.Dataset.zip(ds_tl, ds_tr, ds_bl, ds_br)

# Augment
ds = ds.map(
    ka_layers.vision.Mosaic(
        mosaic_size, offset=(0.25, 0.75), padding_value=114, dtype="uint8"
    )
)
ds = ds.map(
    ka_layers.vision.RandomAffine(
        translate=0.05, scale=0.25, padding_value=114, dtype="uint8"
    )
)
ds = ds.map(ka_layers.vision.CenterCrop(size, dtype="uint8"))
ds = ds.map(ka_layers.vision.RandomGrayscale(p=0.01))
ds = ds.map(ka_layers.vision.RandomHSV(hue=0.015, saturation=0.7, value=0.4))
ds = ds.map(ka_layers.vision.RandomFlip(mode="horizontal"))

# Make gif
images = []
for x in ds.take(1):
    drawed_images = visualization.draw_segmentation_masks(
        x["images"], x["segmentation_masks"], num_classes=2, alpha=0.5
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

import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import ResizeByLongestSide
from keras_aug.utils import augmentation_utils


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    # max_size can be a list which might be useful to perform multi-scale
    # learning
    layer = ResizeByLongestSide(max_size=[256], bounding_box_format="xyxy")
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xyxy")

    data = next(iter(result))
    heights, widths = augmentation_utils.get_images_shape(data["images"])
    shapes = tf.concat([heights, widths], axis=-1)
    larget_sides = tf.reduce_max(shapes, axis=-1)
    print("larget_sides", larget_sides)


if __name__ == "__main__":
    main()

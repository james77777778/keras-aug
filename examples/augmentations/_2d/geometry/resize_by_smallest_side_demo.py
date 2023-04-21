import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation import ResizeBySmallestSide
from keras_aug.utils import augmentation


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    # min_size can be a list which might be useful to perform multi-scale
    # learning
    layer = ResizeBySmallestSide(min_size=[256], bounding_box_format="xyxy")
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xyxy")

    data = next(iter(result))
    heights, widths = augmentation.get_images_shape(data["images"])
    shapes = tf.concat([heights, widths], axis=-1)
    smaller_sides = tf.reduce_min(shapes, axis=-1)
    print("smaller_sides", smaller_sides)


if __name__ == "__main__":
    main()

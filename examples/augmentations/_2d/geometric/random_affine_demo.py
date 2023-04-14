import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import RandomAffine


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=1
    )
    layer = RandomAffine(
        rotation_factor=(-0.2, 0.2),
        translation_height_factor=(-0.2, 0.2),
        translation_width_factor=(-0.2, 0.2),
        zoom_height_factor=(-0.2, 0.2),
        zoom_width_factor=(-0.2, 0.2),
        shear_height_factor=(-0.2, 0.2),
        shear_width_factor=(-0.2, 0.2),
        fill_value=114,
        bounding_box_format="xyxy",
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data_across_batch(result, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()

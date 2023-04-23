import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation import RandomAffine


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=9
    )

    # The bounding boxes after rotation augmentation is not optimal,
    # so you should not augment rotation by large value.
    # 10 degrees should be fine.
    layer = RandomAffine(
        rotation_factor=10,  # 10 degrees
        translation_height_factor=0.2,  # 20%
        translation_width_factor=0.2,
        zoom_height_factor=0.2,  # 20%
        zoom_width_factor=0.2,
        shear_height_factor=0.2,  # 20%
        shear_width_factor=0.2,
        fill_value=114,
        bounding_box_format="xyxy",
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()

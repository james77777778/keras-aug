import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import RandomAffine


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=9
    )
    # RandomAffine does not currently support bounding boxes augmentation with
    # zoom factors.
    layer = RandomAffine(
        rotation_factor=10 / 360,  # 20 degrees
        translation_height_factor=0.1,  # 20%
        translation_width_factor=0.1,
        zoom_height_factor=0.0,
        zoom_width_factor=0.0,
        shear_height_factor=0.1,  # 20%
        shear_width_factor=0.1,
        fill_value=114,
        bounding_box_format="xyxy",
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()

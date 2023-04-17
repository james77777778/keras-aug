import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import RandomBrightnessContrast


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = RandomBrightnessContrast(
        value_range=(0, 255),
        brightness_factor=(0.5 - 0.2, 0.5 + 0.2),  # 40%
        contrast_factor=(0.5 - 0.2, 0.5 + 0.2),  # 40%
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )


if __name__ == "__main__":
    main()

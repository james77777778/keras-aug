import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import MosaicYOLOV8


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=9
    )
    layer = MosaicYOLOV8(
        target_size=(640, 640), padding_value=114, bounding_box_format="xyxy"
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )


if __name__ == "__main__":
    main()

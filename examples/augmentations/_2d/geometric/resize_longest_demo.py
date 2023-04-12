import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import ResizeLongest


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=1
    )
    layer = ResizeLongest(max_size=[64, 128, 256], bounding_box_format="xyxy")
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data_across_batch(result, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()

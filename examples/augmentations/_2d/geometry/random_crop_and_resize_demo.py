import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import RandomCropAndResize


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=9
    )
    layer = RandomCropAndResize(
        height=256,
        width=256,
        crop_area_factor=(0.5, 0.5),
        aspect_ratio_factor=(0.5, 0.5),
        bounding_box_format="xyxy",
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()

import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_aug import layers as ka_layers

BATCH_SIZE = 64
NUM_CLASSES = 3
INPUT_SIZE = (128, 128)

# Create a `tf.data.Dataset`-compatible preprocessing pipeline with all backends
train_dataset, validation_dataset = tfds.load(
    "rock_paper_scissors", as_supervised=True, split=["train", "test"]
)
train_dataset = (
    train_dataset.batch(BATCH_SIZE)
    .map(
        lambda images, labels: {
            "images": tf.cast(images, "float32") / 255.0,
            "labels": tf.one_hot(labels, NUM_CLASSES),
        }
    )
    .map(ka_layers.vision.Resize(INPUT_SIZE))
    .shuffle(128)
    .map(ka_layers.vision.RandAugment())
    .map(ka_layers.vision.CutMix(num_classes=NUM_CLASSES))
    .map(lambda data: (data["images"], data["labels"]))
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset = (
    validation_dataset.batch(BATCH_SIZE)
    .map(
        lambda images, labels: {
            "images": tf.cast(images, "float32") / 255.0,
            "labels": tf.one_hot(labels, NUM_CLASSES),
        }
    )
    .map(ka_layers.vision.Resize(INPUT_SIZE))
    .map(lambda data: (data["images"], data["labels"]))
    .prefetch(tf.data.AUTOTUNE)
)

# Create a CNN model
model = keras.models.Sequential(
    [
        keras.Input((*INPUT_SIZE, 3)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(256, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(),
    metrics=["accuracy"],
)

# Train your model
model.fit(train_dataset, validation_data=validation_dataset, epochs=8)

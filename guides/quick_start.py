import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_aug import layers as ka_layers

BATCH_SIZE = 64
NUM_CLASSES = 3
INPUT_SIZE = (128, 128)

# Create a `tf.data.Dataset`-compatible preprocessing pipeline.
# Note that this example works with all backends.
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
    .map(ka_layers.vision.Rescale(scale=2.0, offset=-1))  # [0, 1] to [-1, 1]
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
    .map(ka_layers.vision.Rescale(scale=2.0, offset=-1))  # [0, 1] to [-1, 1]
    .map(lambda data: (data["images"], data["labels"]))
    .prefetch(tf.data.AUTOTUNE)
)

# Create a model using MobileNetV2 as the backbone.
backbone = keras.applications.MobileNetV2(
    input_shape=(*INPUT_SIZE, 3), include_top=False
)
backbone.trainable = False
inputs = keras.Input((*INPUT_SIZE, 3))
x = backbone(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    metrics=["accuracy"],
)

# Train and evaluate your model
model.fit(train_dataset, validation_data=validation_dataset, epochs=8)
model.evaluate(validation_dataset)

from keras_aug._src.keras_aug_export import keras_aug_export

__version__ = "1.0.1"


@keras_aug_export("keras_aug")
def version():
    return __version__

from keras_aug._src.keras_aug_export import keras_aug_export

__version__ = "0.9.0"


@keras_aug_export("kimm")
def version():
    return __version__

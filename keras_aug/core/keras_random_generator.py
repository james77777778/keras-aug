try:
    import keras.src.backend

    KerasRandomGenerator = keras.src.backend.RandomGenerator

except ModuleNotFoundError:
    """Backwards compatibility for keras<=2.12"""
    import keras.backend

    keras.backend.RandomGenerator
    KerasRandomGenerator = keras.backend.RandomGenerator

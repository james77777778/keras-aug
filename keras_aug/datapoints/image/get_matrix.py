import tensorflow as tf
from tensorflow import keras


def get_rotation_matrix(
    angles, image_height, image_width, to_square=False, name=None
):
    """Returns projective transforms for the given angles.

    Args:
        angles (tf.Tensor): a vector with the angles to rotate each image
            in the batch.
        image_height (tf.Tensor): Height of the images to be transformed.
        image_width (tf.Tensor): Width of the images to be transformed.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """
    with keras.backend.name_scope(name or "rotation_matrix"):
        x_offset = (image_width - 1) - (
            tf.cos(angles) * (image_width - 1)
            - tf.sin(angles) * (image_height - 1)
        )
        x_offset /= 2.0
        y_offset = (image_height - 1) - (
            tf.sin(angles) * (image_width - 1)
            + tf.cos(angles) * (image_height - 1)
        )
        y_offset /= 2.0
        num_angles = tf.shape(angles)[0]
        matrix = tf.concat(
            [
                tf.cos(angles),
                -tf.sin(angles),
                x_offset,
                tf.sin(angles),
                tf.cos(angles),
                y_offset,
                tf.zeros((num_angles, 2), angles.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_angles, 1), angles.dtype)], axis=1
            )
            matrix = tf.reshape(matrix, (num_angles, 3, 3))
        return matrix


def get_translation_matrix(
    translations, image_height, image_width, to_square=False, name=None
):
    """Returns projective transforms for the given translations.

    Args:
        translations (tf.Tensor): A matrix of 2-element lists representing
            ``[dx, dy]`` to translate for a batch of images.
        image_height (tf.Tensor): Height of the images to be transformed.
        image_width (tf.Tensor): Width of the images to be transformed.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """
    with keras.backend.name_scope(name or "translation_matrix"):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        matrix = tf.concat(
            values=[
                tf.ones((num_translations, 1), translations.dtype),
                tf.zeros((num_translations, 1), translations.dtype),
                -translations[:, 0, tf.newaxis] * image_width,
                tf.zeros((num_translations, 1), translations.dtype),
                tf.ones((num_translations, 1), translations.dtype),
                -translations[:, 1, tf.newaxis] * image_height,
                tf.zeros((num_translations, 2), translations.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_translations, 1), translations.dtype)],
                axis=1,
            )
            matrix = tf.reshape(matrix, (num_translations, 3, 3))
        return matrix


def get_zoom_matrix(
    zooms, image_height, image_width, to_square=False, name=None
):
    """Returns projective transforms for the given zooms.

    Args:
        zooms (tf.Tensor): A matrix of 2-element lists representing
            ``[zx, zy]`` to zoom for a batch of images.
        image_height (tf.Tensor): Height of the images to be transformed.
        image_width (tf.Tensor): Width of the images to be transformed.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """
    with keras.backend.name_scope(name or "zoom_matrix"):
        num_zooms = tf.shape(zooms)[0]
        # The zoom matrix looks like:
        #     [[zx 0 0]
        #      [0 zy 0]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Zoom matrices are always float32.
        x_offset = ((image_width - 1.0) / 2.0) * (1.0 - zooms[:, 0, tf.newaxis])
        y_offset = ((image_height - 1.0) / 2.0) * (
            1.0 - zooms[:, 1, tf.newaxis]
        )
        matrix = tf.concat(
            values=[
                zooms[:, 0, tf.newaxis],
                tf.zeros((num_zooms, 1), zooms.dtype),
                x_offset,
                tf.zeros((num_zooms, 1), zooms.dtype),
                zooms[:, 1, tf.newaxis],
                y_offset,
                tf.zeros((num_zooms, 2), zooms.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_zooms, 1), zooms.dtype)], axis=1
            )
            matrix = tf.reshape(matrix, (num_zooms, 3, 3))
        return matrix


def get_shear_matrix(shears, to_square=False, name=None):
    """Returns projective transforms for the given shears.

    Args:
        shears (tf.Tensor): A matrix of 2-element lists representing `[sx, sy]`
            to shear for a batch of images.
        to_square (bool, optional): Whether to append ones to last dimension
            and reshape to ``(batch_size, 3, 3)``. Defaults to ``False``.
        name (str, optional): The name of the op. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """
    with keras.backend.name_scope(name or "shear_matrix"):
        num_shears = tf.shape(shears)[0]
        # The transform matrix looks like:
        # (1, x, 0)
        # (y, 1, 0)
        # (0, 0, 1)
        # where the last entry is implicit.
        matrix = tf.concat(
            values=[
                tf.ones((num_shears, 1), shears.dtype),
                shears[:, 0, tf.newaxis],
                tf.zeros((num_shears, 1), shears.dtype),
                shears[:, 1, tf.newaxis],
                tf.ones((num_shears, 1), shears.dtype),
                tf.zeros((num_shears, 3), shears.dtype),
            ],
            axis=1,
        )
        if to_square:
            matrix = tf.concat(
                [matrix, tf.ones((num_shears, 1), shears.dtype)], axis=1
            )
            matrix = tf.reshape(matrix, (num_shears, 3, 3))
        return matrix

import tensorflow as tf


def stateless_random_beta(
    shape, seed_alpha, seed_beta, alpha, beta, dtype=tf.float32
):
    seed_alpha = tf.cast(seed_alpha, dtype=tf.int32)
    seed_beta = tf.cast(seed_beta, dtype=tf.int32)
    sampled_alpha = tf.random.stateless_gamma(
        shape, seed_alpha, alpha=alpha, dtype=dtype
    )
    sampled_beta = tf.random.stateless_gamma(
        shape, seed_beta, alpha=beta, dtype=dtype
    )
    return sampled_alpha / (sampled_alpha + sampled_beta)


def stateless_random_dirichlet(shape, seed, alpha, dtype=tf.float32):
    sampled_gamma = tf.random.stateless_gamma(
        shape, seed, alpha=alpha, dtype=dtype
    )
    return sampled_gamma / tf.reduce_sum(sampled_gamma, axis=-1, keepdims=True)

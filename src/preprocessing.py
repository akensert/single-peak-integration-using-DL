import tensorflow as tf


def interpolate(x, target_size):
    x = tf.image.resize(x[:, tf.newaxis, :], (target_size, 1))
    return tf.squeeze(x, axis=1)

def normalize(x):
    return x / tf.reduce_max(x)
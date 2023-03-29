import tensorflow as tf
import numpy as np

import preprocessing



def Dataset(
    x, 
    y, 
    sample_weight=None, 
    batch_size=32, 
    training=False, 
    augmentation=None, 
    shuffle_buffer_size=2048
):
    if sample_weight is None:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
    if training:
        dataset = dataset.shuffle(shuffle_buffer_size)
        if augmentation is not None:
            dataset = dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_train_data(files, target_signal_size):
    x, y = [], []
    for time, signal, label in zip(*files):
        signal = np.expand_dims(signal, -1).astype(np.float32)
        label = label.astype(np.float32)
        signal = preprocessing.interpolate(signal, target_signal_size).numpy()
        signal = preprocessing.normalize(signal).numpy()
        x.append(signal)
        y.append(label / time.max()) # normalize between [0, 1]
    return np.stack(x, axis=0), np.stack(y, axis=0)


import tensorflow as tf
from tensorflow import keras

import preprocessing



def ConvolutionalNet(
    input_shape=(512, 1),
    filters=[64, 128, 128, 256, 256],
    kernel_size=9,
    pool_size=2,
    pool_type='avg',
    units=2048,
    dropout=0.5,
    activation='relu',
    batch_norm=False,
):

    inputs = keras.layers.Input(input_shape, dtype='float32')

    x = inputs

    # Perform convolutional steps of signal to get encoding: Encoder part of model
    for f in filters:
        if pool_type == 'conv':
            x = keras.layers.Conv1D(
                f, kernel_size, pool_size, padding='same', activation=None)(x)
            if batch_norm:
                x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation(activation)(x)

        else:
            x = keras.layers.Conv1D(f, kernel_size, 1, padding='same', activation=None)(x)
            if batch_norm:
                x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation(activation)(x)

            if pool_type == 'max':
                x = keras.layers.MaxPooling1D(pool_size)(x)
            else:
                x = keras.layers.AveragePooling1D(pool_size)(x)

    # Flatten encoding and pass through fully-connected network: Prediction part of model
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(units, activation)(x)

    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(units, activation)(x)

    x = keras.layers.Dense(2, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=x)


class ServeModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], tf.float32),
            tf.TensorSpec([], tf.int32)
        ]
    )
    def __call__(self, signal, n=1):
        # add 'channel' dim (needed for cnn layers)
        signal = tf.expand_dims(signal, -1)
        # interpolate to desired size and normalize between [0, 1]
        signal = preprocessing.interpolate(signal, self.model.input_shape[1])
        signal = preprocessing.normalize(signal)
        # add batch dim (needed for model)
        x =  tf.expand_dims(tf.transpose(tf.tile(signal, [1, n])), -1)
        # pass signal to model and predict begin and end cut of peak.
        prediction = self.model(x, training=True)
        return {
            'mean': tf.math.reduce_mean(prediction, axis=0),
            'std': tf.math.reduce_std(prediction, axis=0)
        }
    
    def export(self, path):
        tf.saved_model.save(self, path)
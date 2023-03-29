import tensorflow as tf
from abc import ABCMeta, abstractmethod

import preprocessing


class Compose:

    def __init__(self, augmentations):
        self.augmentations = augmentations

    @tf.function
    def __call__(self, x, y):
        for augmentation in self.augmentations:
            x, y = augmentation(x, y)
        return x, y


class Augmentation(metaclass=ABCMeta):

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, x, y):
        y = tf.cast(y, tf.float32)
        x = tf.cast(x, tf.float32)
        tf.debugging.assert_rank(x, 2, message='Input x needs to be of rank 2.')
        condition = tf.random.uniform(()) > self.prob
        if condition:
            return x, y 
        return self.call(x, y)

    @abstractmethod
    def call(self, x, y):
        pass


class RandomNoise(Augmentation):

    def __init__(self, prob, stddev_range=(0.001, 0.025)):
        super().__init__(prob)
        self.stddev_range = stddev_range

    def call(self, x, y):
        stddev = tf.random.uniform((), *self.stddev_range)
        stddev = tf.expand_dims(stddev, -1)
        noise = tf.random.normal(tf.shape(x)) * stddev
        noise = tf.cast(noise, x.dtype)
        x += noise
        x = preprocessing.normalize(x)
        return x, y


class RandomReverse(Augmentation):

    def call(self, x, y):
        return x[::-1], (1 - y)[::-1]
    

class RandomCropping(Augmentation):

    def call(self, x, y):
        begin = tf.random.uniform((), 0, y[0])
        end = tf.random.uniform((), y[1], 1.0)
        x = tf.image.crop_and_resize(
            x[tf.newaxis, :, :, tf.newaxis], 
            boxes=[[begin, 0.0, end, 1.0]],
            box_indices=[0],
            crop_size=[tf.shape(x)[0], 1])
        x = tf.squeeze(x, axis=[0, 3])
        y = (y - begin) / (end - begin)
        return x, y


class RandomPadding(Augmentation):

    def __init__(self, prob, padding_factor_range=(0, 6)):
        super().__init__(prob)
        self.padding_factor_range = padding_factor_range

    def call(self, x, y):

        pfr = self.padding_factor_range
        size = tf.cast(tf.shape(x)[0], tf.float32)

        begin, end = tf.math.round(y[0] * size), tf.math.round(y[1] * size)
        
        pad_left = tf.random.uniform((), pfr[0] * size, pfr[1] * size)
        x = tf.pad(x, [(tf.cast(pad_left, tf.int32), 0), (0, 0)], mode='CONSTANT', constant_values=x[0, 0])
        
        pad_right = tf.random.uniform((), pfr[0] * size, pfr[1] * size)
        x = tf.pad(x, [(0, tf.cast(pad_right, tf.int32)), (0, 0)], mode='CONSTANT', constant_values=x[-1, 0])
        
        new_size = tf.cast(tf.shape(x)[0], tf.float32)

        ratio = size / new_size

        x = preprocessing.interpolate(x, size)
        
        new_begin = (begin + pad_left) * ratio
        new_end = (end + pad_left) * ratio

        y = tf.concat([[new_begin], [new_end]], axis=0) / size
        return x, y


class RandomBaselineDrift(Augmentation):

    def __init__(self, prob, multiplier_range=(-0.025, 0.025), num_distortions=10):
        super().__init__(prob)
        self.multiplier_range = multiplier_range
        self.num_distortions = num_distortions

    def call(self, x, y):
        t = tf.expand_dims(tf.cast(tf.linspace(-1, 1, tf.shape(x)[0]), tf.float32), -1)
        baseline_drift = tf.zeros(tf.shape(x), dtype='float32')
        n = self.num_distortions
        for _ in tf.range(n):
            multiplier = tf.random.uniform((), *self.multiplier_range)
            a = tf.random.uniform((), -20, 20)
            b = tf.random.uniform((), -20, 20)
            baseline_drift += self._sigmoid(t, a, b, multiplier) / n
        x = x + baseline_drift
        x = preprocessing.normalize(x)
        return x, y

    @staticmethod
    def _sigmoid(t, a, b, multiplier):
        return 1 / (1 + tf.math.exp( - (t * a + b) )) * multiplier

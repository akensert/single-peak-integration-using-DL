import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import glob
import sys
import os
import argparse

sys.path.append('../../src/')

import augmentation
import model
import dataset


def read_excel(excel_file):
    data = pd.read_excel(excel_file, header=None)
    time = data[0].values.astype(np.float32)
    signal = data[1].values.astype(np.float32)
    label = data.iloc[0, 2:4].values.astype(np.float32)
    return time, signal, label

def read_files(path, num_processes=None):
    files = glob.glob(path)
    # indices = np.argsort([
    #     int(f.split('.xlsx')[0].split('/')[-1]) for f in files])
    # files = np.array(files)[indices]
    with multiprocessing.Pool(num_processes) as pool:
        data = list(
            tqdm(
                pool.imap(read_excel, files), 
                total=len(files), 
                desc='Reading files'
            )
        )
    times, signals, labels = list(zip(*data))
    return times, signals, labels, files



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cnn_model/', type=str)
    parser.add_argument('--train_files', default='train_files/', type=str)
    parser.add_argument('--signal_size', default=512, type=int)
    args = parser.parse_args()

    MODEL_PATH = os.path.join('../models/', args.model)
    FILES_PATH = os.path.join('../inputs/', args.train_files, '*')
    TARGET_SIGNAL_SIZE = args.signal_size


    times, signals, labels, _ = read_files(FILES_PATH)

    x, y = dataset.get_train_data(
        [times, signals, labels], TARGET_SIGNAL_SIZE)

    train_data = dataset.Dataset(
        x, 
        y, 
        training=True, 
        augmentation=augmentation.Compose([
            augmentation.RandomBaselineDrift(prob=0.5),
            augmentation.RandomCropping(prob=0.5),
            augmentation.RandomPadding(prob=0.5),
            augmentation.RandomNoise(prob=0.5),
            augmentation.RandomReverse(prob=0.5)
        ]), 
    )

    cnn_model = model.ConvolutionalNet((TARGET_SIGNAL_SIZE, 1))

    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1e-3, 
                decay_steps=10_000, 
                end_learning_rate=1e-6
            )
        ), 
        loss=tf.keras.losses.BinaryCrossentropy()
    )

    cnn_model.fit(train_data, epochs=300, verbose=2)

    model.ServeModel(cnn_model).export(MODEL_PATH)

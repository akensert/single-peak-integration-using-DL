import tensorflow as tf
import numpy as np
import glob
import os
import sys
import re

sys.path.append('../../src/')

import postprocessing


def read_txt_file(path):
    time, signal = [], []
    with open(path, 'rb') as fh:
        contents = fh.read()
        contents = contents.decode('utf-8', errors='replace')
        contents = contents.split('Value(mAU)')[-1].split('\n')[1:-1]
        for line in contents:
            line = line.strip().split('\t')
            time.append(float(line[0].replace(',', '.')))
            signal.append(float(line[-1].replace(',', '.')))
    return np.array(time), np.array(signal)

def get_file_names(path):
    files = glob.glob(os.path.join(path, '*'))
    files = [os.path.abspath(file.replace('\\', '/')) for file in files]
    names = [os.path.basename(f.rstrip('.txt')) for f in files]
    try:
        regex = re.compile(r'\d+')
        file_ids = np.argsort([int(regex.findall(name)[-1]) for name in names])
    except:
        file_ids = np.argsort(names)
    files = np.array(files)[file_ids]
    names = np.array(names)[file_ids]
    return files, names

def get_figures_path(path, file_path, format='svg'):
    return os.path.abspath(path.rstrip('/')) + '/' + file_path.split('/')[-1].split('.txt')[0] + f'.{format}'


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='cnn_model/', type=str)
    parser.add_argument('--files', default='files/', type=str)
    parser.add_argument('--figures', default='figures/', type=str)
    parser.add_argument('--figure_format', default='svg', type=str)
    parser.add_argument('--results', default='results.csv', type=str)
    parser.add_argument('--csv_sep', default='\t', type=str)

    args = parser.parse_args()

    MODEL_PATH = os.path.join('../models/', args.model)
    FILES_PATH = os.path.join('../inputs/', args.files)
    FIGURES_PATH = os.path.join('../outputs/', args.figures)
    RESULT_PATH = os.path.join('../outputs/', args.results)
    FIGURE_SAVE_FORMAT = args.figure_format
    SEP = args.csv_sep


    file_paths, file_names = get_file_names(FILES_PATH)

    loaded_model = tf.saved_model.load(MODEL_PATH)

    if not os.path.exists(RESULT_PATH):
        os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
        write_header = True
    else:
        write_header = False

    with open(RESULT_PATH, 'a') as fh:

        if write_header:
            fh.write(
                f'name{SEP}' +
                f'begin{SEP}' + 
                f'end{SEP}' +
                f'zeroth_moment{SEP}' +
                f'first_moment{SEP}' +
                f'second_moment{SEP}' + 
                f'third_moment{SEP}' + 
                f'second_moment_w0.5{SEP}' + 
                f'file{SEP}' +
                'figure' +
                '\n'
            )

        for file_path, file_name in zip(file_paths, file_names):
            
            figures_path = get_figures_path(FIGURES_PATH, file_path, FIGURE_SAVE_FORMAT)

            time, signal = read_txt_file(file_path)

            pred = loaded_model(signal, n=30)

            stats, peak = postprocessing.compute_stats(pred, time, signal)

            postprocessing.plot(time, signal, peak, save_path=figures_path)

            fh.write(
                f'{file_name}{SEP}' +
                f'{peak.begin}{SEP}' +
                f'{peak.end}{SEP}' +
                f'{stats.zeroth_moment}{SEP}' +
                f'{stats.first_moment}{SEP}' +
                f'{stats.second_moment}{SEP}' +
                f'{stats.third_moment}{SEP}' +
                f'{stats.second_moment_v2}{SEP}' +
                f'{file_path}{SEP}' + 
                f'{figures_path}' +
                '\n'
            )

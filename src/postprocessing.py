import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import os

Stats = namedtuple(
    'Stats', [
        'zeroth_moment', 
        'first_moment', 
        'second_moment',
        'third_moment', 
        'second_moment_v2'
    ]
)

Peak = namedtuple(
    'Peak', [
        'begin', 
        'end', 
        'x', 
        'baseline_y', 
        'signal_y',
    ]
)

def compute_stats(prediction, time, signal):
    
    prediction['mean'] = prediction['mean'].numpy()
    prediction['std'] = prediction['std'].numpy()
    
    # extract predicted peak region
    begin_index = round(prediction['mean'][0] * len(time))
    end_index = round(prediction['mean'][1] * len(time))
    y = signal[begin_index: end_index]
    x = time[begin_index: end_index]

    index_apex = np.where(signal == y.max())[0][0]

    begin_left_edge_index = round((prediction['mean'][0] - prediction['std'][0] * 2) * len(time))
    begin_right_edge_index = round((prediction['mean'][0] + prediction['std'][0] * 2) * len(time))
    end_left_edge_index = round((prediction['mean'][1] - prediction['std'][1] * 2) * len(time))
    end_right_edge_index = round((prediction['mean'][1] + prediction['std'][1] * 2) * len(time))

    y_s = np.median(signal[begin_left_edge_index : begin_right_edge_index])
    y_e = np.median(signal[end_left_edge_index : end_right_edge_index])
    x_s = x[0]
    x_e = x[-1]
    
    # correct baseline: calculate baseline and subtract from chromatogram
    slope = (y_s - y_e) / (x_s - x_e)
    intercept = (x_s * y_e - x_e * y_s) / (x_s - x_e)
    baseline = (slope * x + intercept)
    y_corrected = y - baseline

    stats = Stats(
        zeroth_moment=_compute_moment(x, y_corrected, n=0),
        first_moment=_compute_moment(x, y_corrected, n=1),
        second_moment=_compute_moment(x, y_corrected, n=2),
        third_moment=_compute_moment(x, y_corrected, n=3),
        second_moment_v2=_compute_second_moment(x, y_corrected)
    )

    peak = Peak(
        begin=time[begin_index],
        end=time[end_index],
        x=x, 
        baseline_y=baseline,
        signal_y=y
    )
    
    return stats, peak

def _compute_moment(time, signal, n):
    dt = time[1] - time[0]
    zeroth_moment = np.sum(signal * dt)
    if n == 0:
        return zeroth_moment
    first_moment = np.sum(time * signal * dt) / zeroth_moment
    if n == 1:
        return first_moment
    return np.sum((time - first_moment) ** n * signal * dt) / zeroth_moment

def _compute_second_moment(time, signal):
    'Computes second moment based on width at half height.'
    
    signal_apex = signal.max()

    index_apex = np.where(signal == signal_apex)[0][0]

    try:
        index_start = np.argmin(
            np.abs((signal_apex/2) - signal[:index_apex]))
        
        index_end = np.argmin(
            np.abs((signal_apex/2) - signal[index_apex:])) + len(signal[:index_apex])

        width = time[index_end] - time[index_start]
        return (width**2) / 5.545
    except Exception as e:
        print(e)
        return 'NaN'
    
def plot(time, signal, peak, save_path=None):

    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(time, signal, color='C0', linewidth=1, label='Chromatogram')

    ax.axvline(
        peak.begin, 
        linewidth=1,
        color='C3', 
        label='Peak region')
    
    ax.axvline(
        peak.end, 
        linewidth=1,
        color='C3')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel("Signal (mAU)")
    ax.set_xlabel("Time (min)")

    ax.plot(
        peak.x, 
        peak.baseline_y, 
        color='black',
        linewidth=1,
        linestyle='--',
        label='Baseline')

    plt.fill_between(
        x=peak.x, 
        y1=peak.baseline_y, 
        y2=peak.signal_y,
        color='C3',
        alpha=0.4)

    ax.legend(frameon=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close('all')
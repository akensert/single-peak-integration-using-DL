# Peak integration powered by deep learning

## About
Single chromatographic peaks are extracted using convolutional neural networks; variances of the peaks can then be calcualted, and subsequently used to estimate protein diffusion coefficients.

<img src="https://github.com/akensert/dl-peak-integration/blob/main/media/model-overview.jpg" alt="model-overview" width="800">

## Requriements
- Python ~= 3.10
- Pip ~= 22.0.2
- Python packages, see `requirements.txt`

## Usage

### Training
Add chromatograms (containing single peaks) in `automatic_peak_integration/inputs/train_files/`. Each chromatogram should be an Excel file with four columns: time, signal, \[only first row\] start time of peak, \[only first row\] end time of peak.

Then navigate to `automatic_peak_integration/scripts/` and run, from terminal: `python train.py`.

When training is done, model is saved in `automatic_peak_integration/models/`.

### Predicting (integrating peaks)
Add chromatograms (containing single peaks) in `automatic_peak_integration/inputs/files/`. Each chromatogram should be a txt file with two columns (tab separated): time, signal.

Then navigate to `automatic_peak_integration/scripts/` and run, from terminal: `python integrate.py`

When finished, output is saved in `automatic_peak_integration/outputs/`.
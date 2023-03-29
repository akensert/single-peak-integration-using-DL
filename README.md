# Single-peak extraction with deep learning

## About
Minimalistic and basic implementation to integrate peaks in single-peak chromatograms via deep learning (CNNs). Speeds up experiments for some chromatographers working on e.g. predicing protein diffusion coefficients. Manually integrating peaks (for thousands of chromatograms) is tedious.

## Status
Almost complete but can be further improved in the future.

## Requriements
- Python ~= 3.10.
- Pip ~= 22.0.2
- Python packages, see `requirements.txt`

## Usage

### Training
Navigate to `automatic_peak_integration/scripts/` and run, from terminal: `python train.py`

### Predicting (integrating peaks)
Navigate to `automatic_peak_integration/scripts/` and run, from terminal: `python integrate.py`
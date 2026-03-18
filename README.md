# EEG-Signal-Processing-and-Anomaly-Detection-Using-Python

This is a small project where I tried to detect whether an EEG signal is normal or abnormal using basic signal processing.
The idea is simple:
- Read EEG data from a CSV file
- Clean the signal using a bandpass filter
- Extract some basic features
- Train a model and check if the signal looks abnormal


## What this project does

- Takes EEG data from `.csv`
- Applies filtering (0.5–50 Hz)
- Splits signal into small segments
- Extracts features like:
  - Mean, variance, std
  - Frequency band powers
  - Entropy
- Uses Random Forest to classify
- Gives final result:
  - EEG is NORMAL
  - EEG is ABNORMAL




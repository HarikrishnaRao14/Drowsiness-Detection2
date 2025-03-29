
# Drowsiness Detection System 👁️🚨

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)
![Keras](https://img.shields.io/badge/Keras-2.3%2B-red)

A real-time computer vision system that detects drowsiness using eye state monitoring and triggers preventive alarms.

## Table of Contents
- [Features](#features-)
- [Installation](#installation-)
- [Usage](#usage-)
- [Technical Details](#technical-details-)
- [Troubleshooting](#troubleshooting-)
- [License](#license-)

## Features ✨
- Real-time face and eye detection
- CNN-based eye state classification (Open/Closed)
- Visual and audio alerts when drowsiness detected
- Simple GUI with start/stop controls
- Adjustable sensitivity parameters

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drowsiness-detector.git
cd drowsiness-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required Haar cascades (if not included):
- [haarcascade_frontalface_alt.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml)
- [haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

## Usage 🚀
```bash
python drowsiness_detection.py
```

**Controls:**
- 🟢 Start Button: Begin monitoring
- 🔴 Stop Button: End monitoring
- Press 'q' to quit during operation

**Visual Feedback:**
- 🔵 Blue box: Detected face
- 🟢 Green box: Left eye
- 🟡 Yellow box: Right eye
- 🔴 Red border: Drowsiness alert


## Technical Details 🔧
**Detection Pipeline:**
1. Face detection using Haar cascades
2. Eye region extraction
3. Eye state classification with CNN
4. Drowsiness scoring system
5. Alert triggering (visual + audio)

**Model Architecture:**
- 3 Convolutional Layers + MaxPooling
- Dropout for regularization
- 128-neuron Dense layer
- Softmax output (Open/Closed)

## Troubleshooting 🐛
| Issue | Solution |
|-------|----------|
| No face detection | Check lighting and camera position |
| Alarm not sounding | Verify `alarm.wav` exists in root dir |
| Low accuracy | Adjust `score_threshold` in code |


Developed with ❤️ by Harikrishna Rao 

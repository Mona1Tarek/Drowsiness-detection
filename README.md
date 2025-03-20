# Drowsiness Detection

## Overview
This project implements a **Drowsiness Detection System** that monitors a driver's eyes to detect signs of drowsiness in real time. It leverages **computer vision and deep learning** to analyze eye states (open/closed) and provide alerts when drowsiness is detected.

## Features
- **Real-time eye state detection** using OpenCV and TensorFlow.
- **Drowsiness alert system** with sound notifications.
- **Works with live webcam feed**.
- **Lightweight and efficient model for quick inference**.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mona1Tarek/Drowsiness-detection.git
   cd Drowsiness-detection
   ```
2. Install required dependencies:
   ```bash
   pip install tensorflow opencv-python numpy playsound
   ```

## Files in the Repository
- `EAR3.py`: Eye Aspect Ratio (EAR) calculation script.
- `alarm2.mp3`: Alarm sound file for drowsiness alert.
- `drowing keypoints number function`: Script for extracting facial keypoints.
- `main.py`: Main script for real-time drowsiness detection.
- `mainMediapipe_samples.py`: Script using Mediapipe for face and eye tracking.

## Usage
### Run Drowsiness Detection
```bash
python main.py
```

1. The script will access the webcam and monitor the user's eyes.
2. If the system detects prolonged eye closure, an **alert sound** will be triggered.
3. Press **'q'** to exit the program.


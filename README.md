# Face Recognition

A real-time face recognition system using OpenCV and FisherFace/LBPH recognition algorithms.

## Features

- **Data Collection**: Capture training images from webcam with automatic face detection
- **Face Recognition**: Real-time face detection and recognition with confidence scoring
- **Multiple Subjects**: Support for training and recognizing multiple individuals
- **Haar Cascade Detector**: Uses pre-trained Haar Cascade classifier for face detection
- **Machine Learning**: Implements FisherFace recognizer (with LBPH fallback)

## Project Structure

```
Face_recognition/
├── create_data.py                          # Data collection script
├── face_recognize.py                       # Real-time recognition engine
├── haarcascade_frontalface_default.xml     # Pre-trained face cascade classifier
└── datasets/
    ├── Elon/                               # Training images for Elon
    └── Satheesh/                           # Training images for Satheesh
```

## Requirements

```
opencv-python
numpy
```

## Installation

```bash
pip install opencv-python numpy
```

## Usage

### 1. Collect Training Data

Run the data collection script to capture 30 images per person:

```bash
python create_data.py
```

- Press 'ESC' to exit early
- Captured images are saved to `datasets/[PersonName]/`
- Face size normalized to 130x100 pixels

### 2. Train and Recognize

Run the recognition script to train the model and start real-time recognition:

```bash
python face_recognize.py
```

- Press 'ESC' to exit
- Recognized faces are displayed with name and confidence score
- Unknown faces are saved to `unknown.jpg` after 100 consecutive detections

## Configuration

**In `face_recognize.py`:**
- `confidence < 800`: Threshold for recognizing a face (lower = stricter)
- Adjust threshold based on your accuracy needs

**In `create_data.py`:**
- `count <= 30`: Number of images to capture per person
- Change `sub_data = 'Seesh'` to the person's name before capturing

## How It Works

1. **Detection**: Uses Haar Cascade to detect faces in video frames
2. **Preprocessing**: Converts detected faces to grayscale and resizes to 130x100
3. **Training**: Trains FisherFace recognizer on labeled training images
4. **Recognition**: Predicts identity and confidence for detected faces in real-time

## Notes

- Requires webcam access
- Better results with 30+ well-lit images per person
- Different lighting conditions improve model robustness
- Confidence threshold may need tuning based on your environment

## License

Open source - free to use and modify

## Author

OpenCV Face Recognition Project

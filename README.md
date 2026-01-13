# Sign language recognition system using Mediapipe Hands
This project implements a complete workflow for American Sign Language (ASL) hand-sign recognition, starting from data preprocessing and model training to TensorFlow Lite conversion and a browser-based real-time inference system using MediaPipe Hands + TensorFlow.js.

## Table of Contents

1. Overview
2. Installation
3. Data Preprocessing
4. Model Training
5. Model Evaluation
6. TensorFlow Lite Conversion
7. TFLite Model Evaluation
8. Web Application  
   - Frontend (HTML)  
   - Inference Logic (JavaScript)
9. Docker Deployment
10. End-to-End Workflow

## Overview

The system recognizes ASL characters by detecting hand landmarks using MediaPipe Hands and classifying them with a trained neural network. The workflow begins with transforming raw images into landmark vectors, continues with training a convolutional model, and ends with lightweight deployment options including TensorFlow Lite and TensorFlow.js.

The straightforward organization of the file structure of the project serves as an outline of its implementation.\
root\
│\
├── preprocessing.py \
├── training.py\
├── testing.py\
├── test_tflite.py\
├── tfliteconversion.py\
│\
├── models/\
│   ├── asl_model_norm.h5\
│   └── asl_model_norm.tflite\
│\
├── workspace/\
│   ├── X.npy\
│   ├── y.npy\
│   ├── minmax_scaler.pkl\
│\
├── webapp/\
│   ├── index.html\
│   ├── app.js\
│   ├── model.json\
│   ├── minmax_scaler.json\
│\
└── Dockerfile\

## Quick Start
1. Clone the repository
   ```
   git clone https://github.com/feniatriestocode/sign-translate.git
   ```
2. Navigate to to the webapp directory
   ```
   cd sign-translate/docs
   ```
4. Start the local Server
   ```
   python3 -m http.server
   ```
   
## End-to-End Workflow

1. Run `preprocessing.py` to generate `X.npy` and `y.npy`.  
2. Train the model using `training.py`.  
3. Evaluate the model with `testing.py`.  
4. Convert the model using `tfliteconversion.py`.  
5. Optionally test the TFLite model using `test_tflite.py`.  
6. Open `webapp/index.html` to run real-time ASL detection in the browser.

## Data Preprocessing

This script processes the ASL alphabet dataset and prepares features for model training. It performs:

- Hand-landmark extraction using MediaPipe Hands. Each detected hand yields 21 landmarks represented as 63 numeric values (x, y, z coordinates).
- Label collection and encoding using `LabelEncoder` and optional one-hot encoding.
- Saving of processed arrays to `workspace/X.npy` and `workspace/y.npy`.

## Model Training

The training script loads `X.npy` and `y.npy`, applies MinMax scaling, and trains a convolution-based neural network designed specifically for structured landmark data. It performs the following tasks:

1. Normalizes the dataset using `MinMaxScaler`.  
   The scaler is exported both as a `.pkl` file for Python and a `.json` file for use in the web application.
2. Splits the dataset into training and validation sets.
3. Defines a CNN model operating on landmark data reshaped to `(21, 3, 1)`.
4. Trains with early stopping to prevent overfitting.
5. Saves the final trained model as `models/asl_model_norm.h5`.

## Model Evaluation

The evaluation script tests the trained `.h5` model using a set of labeled images. It:

- Loads the scaler and trained model.
- Extracts hand landmarks from test images using MediaPipe.
- Scales the extracted features.
- Produces predictions and computes accuracy, precision, recall, and F1 score.


## TensorFlow Lite Conversion


This script converts the trained TensorFlow model (`.h5`) into a TensorFlow Lite model (`.tflite`) using the standard `TFLiteConverter`. The resulting file
is suitable for mobile and embedded deployment.

## TFLite Model Evaluation

This script evaluates the TFLite model using the `tf.lite.Interpreter`. It mirrors the Keras evaluation workflow:

- Loads scaler and test images.
- Extracts and scales hand landmarks.
- Runs inference sample-by-sample using the TFLite interpreter.
- Computes model accuracy.

## Web Application

The project includes a browser-based real-time recognition interface built with TensorFlow.js and MediaPipe Hands.

### Frontend (HTML)

The HTML page provides:

- A webcam feed for live hand tracking  
- A canvas for rendered MediaPipe Hands landmarks  
- A prediction display area  
- A start/stop detection control  
- References to TensorFlow.js and MediaPipe libraries  

### Inference Logic (JavaScript)

The browser application performs:

1. Loading of the TensorFlow.js model (`model.json`).  
2. Hand-landmark detection via MediaPipe Hands for each webcam frame.  
3. Formatting of the landmarks into a 63-value vector.  
4. Prediction through the TensorFlow.js model.  
5. Rendering of the hand skeleton and predicted label.

## Docker Deployment

Used to package the environment required for preprocessing, training, or serving the web application.


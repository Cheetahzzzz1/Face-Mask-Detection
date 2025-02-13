# Face-Mask-Detection

# Overview

This project is a Face Mask Detection System using Convolutional Neural Networks (CNNs) in TensorFlow/Keras.

The model classifies whether a person in an image is wearing a mask or not. It can be extended for a real time detection using OpenCV and deployed on edge devices. 

# Features

1. Detects "Mask" or "No Mask" in images.

2. Trained on a dataset with diverse face images.

3. Uses CNN-based Deep Learning for high accuracy.

4. Implements data augmentation to improve generalization.

5. Supports custom image testing.

6. Can be extended for real-time detection using OpenCV.

# Dataset

We use Face Mask Detection Dataset from GitHub, which consists of :

1. **Train Set** : with_mask & without_mask folders.

2. **Test_Set** : with_mask & without_mask folders.

# Installation & Setup

1. <ins>Clone the Repository</ins>

         git clone https://github.com/your-repo/face-mask-detection.git
         cd face-mask-detection

2.<ins>Install Dependencies</ins>

         pip install tensorflow opencv-python numpy matplotlib

3. <ins>Download Dataset</ins>

         !wget --no-check-certificate \
         "https://github.com/chandrikadeb7/Face-Mask- 
           Detection/archive/refs/heads/master.zip" \
           -O face_mask_dataset.zip

4. <ins>Extract Dataset</ins>

         import zipfile
         with zipfile.ZipFile("face_mask_dataset.zip", 'r') as zip_ref:
         zip_ref.extractall("face_mask_data")

5. <ins>Run Training</ins>

         python train.py

6. <ins> Test the Model</ins>

         python test.py

# Model Architecture

The CNN model consists of:

1. 3 Convolutional Layers with ReLu Activation.

2. MaxPooling Layers for feature extraction.

3. Flatten Layer to convert 2D features to 1D

4. Dense Layers for classification

5. Output Layer with sigmoid activation (binary classification)

# Training and Evaluation

1. Model is trained on 80% of images and tested on 20%.

2. Augmentation applied : Rotation, Zoom, Flip, Shift.

3. Optimizer : Adam

4. Loss Function : Binary Crossentropy

5. Performance Metrics : Accuracy, Precision, Recall

# Future Enhancements

1. Add multi-class classification for detecting incorrect mask usage.

2. Improve accuracy with pre-trained models like MobileNetV2 or ResNet50.

3. Implement real-time alert system for mask violations.

# Acknowledgements

1. Dataset : Chandrika Deb

2. TensorFlow and Keras for deep learning models.

3. OpenCV for image processing

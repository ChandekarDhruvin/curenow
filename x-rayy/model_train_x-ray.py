import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read X-ray as grayscale
    img = cv2.resize(img, (256, 256))  # Resize
    img = img / 255.0  # Normalize
    return img.reshape(256, 256, 1)  # Reshape for CNN

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(2,2),  
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (Normal vs. Pneumonia)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

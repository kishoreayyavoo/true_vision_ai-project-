import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearnex import patch_sklearn
patch_sklearn()
train_dir = 'D:/deepfake_detection_project/data/processed_frames/train'  
validation_dir = 'D:/deepfake_detection_project/data/processed_frames/val'  
def load_data(train_dir, validation_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    validation_data = datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_data, validation_data
def create_model(input_shape=(224, 224, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
def train_model(train_data, validation_data, epochs=10):
    model = create_model()
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data
    )
    model.save('deepfake_model.h5')  
if __name__ == "__main__":
    train_data, validation_data = load_data(train_dir, validation_dir)
    train_model(train_data, validation_data)

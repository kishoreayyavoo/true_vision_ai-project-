import numpy as np
import pandas as pd
import cv2
import os
from mtcnn import MTCNN
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

detector = MTCNN()
def extract_landmarks(image):
    results = detector.detect_faces(image)
    
    if len(results) == 0:
        raise ValueError("No face detected in the image.")
    
    landmarks = results[0]['keypoints']
    landmark_points = [
        landmarks['left_eye'], 
        landmarks['right_eye'], 
        landmarks['nose'], 
        landmarks['mouth_left'], 
        landmarks['mouth_right']
    ]
    
    return np.array(landmark_points).flatten()
def preprocess_data(data_path, label):
    landmarks_list = []
    labels = []
    
    for filename in os.listdir(data_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(data_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                landmarks = extract_landmarks(image_rgb)
                landmarks_list.append(landmarks)
                labels.append(label)
            except ValueError as e:
                print(e)
                continue
    
    return np.array(landmarks_list), np.array(labels)

real_images_path = 'J:/true_vision_ai/true_vision_ai/data/real'
fake_images_path = 'J:/true_vision_ai/true_vision_ai/data/fake'
    

real_landmarks, real_labels = preprocess_data(real_images_path, label=0) 
fake_landmarks, fake_labels = preprocess_data(fake_images_path, label=1)  


X = np.vstack((real_landmarks, fake_landmarks))
y = np.hstack((real_labels, fake_labels))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, -1)  
X_val = X_val.reshape(X_val.shape[0], 1, -1)

def create_gru_model(input_shape):
    model = models.Sequential()
    model.add(layers.GRU(128, input_shape=input_shape, return_sequences=True))
    model.add(layers.GRU(64))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (1, X_train.shape[2])  
model = create_gru_model(input_shape)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save('models/gru_deepfake_model.h5')
print("Model saved as 'models/gru_deepfake_model.h5'.")

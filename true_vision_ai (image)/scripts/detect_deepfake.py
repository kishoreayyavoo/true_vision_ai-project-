import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

model = load_model(os.path.join('models', 'gru_deepfake_model.h5'))

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

def preprocess_landmarks(landmark_array):
    return landmark_array.reshape(1, 1, -1)  
def predict_deepfake(image_path, threshold=0.6):  
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid format: {image_path}.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        landmarks = extract_landmarks(image_rgb)
    except ValueError as e:
        print(e)
        return
    
    input_data = preprocess_landmarks(landmarks)
    
    prediction = model.predict(input_data)
    print(f"Raw prediction score: {prediction[0][0]}")

    if prediction[0][0] >= threshold:
        print(f"The image {image_path} is classified as Fake.")
    else:
        print(f"The image {image_path} is classified as Real.")

if __name__ == "__main__":
    image_path = input("Enter the path to the image you want to classify: ")
    
    predict_deepfake(image_path)

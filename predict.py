import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 
    return img_array
def predict_image(model, img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)
    return prediction[0][0]  
def main():
    model_path = 'D:/deepfake_detection_project/models/saved_models/deepfake_model.h5'  
    img_dir = 'D:/deepfake_detection_project/data/processed_frames/test//fake'  
    model = load_model(model_path)
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            prediction = predict_image(model, img_path)
            if prediction >=0.5:
                print(f"{img_file}: Real (Confidence: {prediction:.2f})")
            else:
                print(f"{img_file}: Fake (Confidence: {prediction:.2f})")
if __name__ == "__main__":
    main()

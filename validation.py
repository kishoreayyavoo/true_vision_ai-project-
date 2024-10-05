import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def load_data(validation_dir, batch_size=32, img_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    validation_data = datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    return validation_data

def validate_model(model_path, validation_data):
    model = tf.keras.models.load_model(model_path)
    evaluation = model.evaluate(validation_data)
    print(f"Validation Loss: {evaluation[0]}")
    print(f"Validation Accuracy: {evaluation[1]}")

def main():
    validation_data = load_data('D:/deepfake_detection_project/data/processed_frames/val')
    validate_model('D:/deepfake_detection_project/models/saved_models/deepfake_model.h5', validation_data)

if __name__ == "__main__":
    main()

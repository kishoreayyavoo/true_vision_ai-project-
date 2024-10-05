import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn_rnn_model import create_model
import json
import os
def load_data(train_dir, validation_dir, batch_size=32, img_size=(224, 224)):
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
def train_model(model, train_data, validation_data, epochs=10):
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data
    )
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    if not os.path.exists('models/saved_models'):
        os.makedirs('models/saved_models')
    model.save('D:/deepfake_detection_project/models/saved_models/deepfake_model.h5')
def main():
    input_shape = (224, 224, 3)  
    num_classes = 1  
    model = create_model(input_shape, num_classes)
    train_data, validation_data = load_data(
        'D:/deepfake_detection_project/data/processed_frames/train',
        'D:/deepfake_detection_project/data/processed_frames/val'
    )
    train_model(model, train_data, validation_data)
if __name__ == "__main__":
    main()

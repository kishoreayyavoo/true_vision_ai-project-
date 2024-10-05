import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.plot_utils import plot_accuracy_loss
def load_data(validation_dir, batch_size=32, img_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    validation_data = datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return validation_data
def evaluate_model(model, validation_data):
    results = model.evaluate(validation_data)
    print(f"Validation Loss: {results[0]}")
    print(f"Validation Accuracy: {results[1]}")
    return results
def main():
    model = load_model('D:\deepfake_detection_project\models\saved_models/deepfake_model.h5')
    validation_data = load_data('D:\deepfake_detection_project\data\processed_frames\val')
    results = evaluate_model(model, validation_data)
    plot_accuracy_loss('training_history.json')
if __name__ == "__main__":
    main()

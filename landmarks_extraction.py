import cv2
import dlib
import pandas as pd
import os
predictor_path = "D:/deepfake_detection_project/models/shape_predictor_68_face_landmarks.dat"  # Path to landmark model
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor(predictor_path)  
def extract_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot load image: {img_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]
        return landmarks_points
    return None
def save_landmarks_to_csv(landmarks_dict, csv_path):
    df = pd.DataFrame.from_dict(landmarks_dict, orient='index')
    df.to_csv(csv_path, index_label='Image')
def process_images_in_directory(img_dir, output_csv):
    landmarks_dict = {}
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks:
                flattened_landmarks = [coord for point in landmarks for coord in point]
                landmarks_dict[img_file] = flattened_landmarks
            else:
                print(f"No landmarks detected for {img_file}")
    save_landmarks_to_csv(landmarks_dict, output_csv)
def main():
    img_dir = 'D:/deepfake_detection_project/data/processed_frames/test/fake'  # Update this path to your directory (train/test/val)
    output_csv = 'landmarks_output.csv' 
    process_images_in_directory(img_dir, output_csv)
    print(f"Landmarks have been saved to {output_csv}")
if __name__ == "__main__":
    main()

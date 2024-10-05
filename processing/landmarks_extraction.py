import cv2
import dlib
import os
def extract_landmarks_from_frames(frame_dir, predictor_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    for frame_file in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_file)
        img = cv2.imread(frame_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        output_frame_path = os.path.join(output_dir, f"landmark_{frame_file}")
        cv2.imwrite(output_frame_path, img)
if __name__ == "__main__":
    frame_dir = 'D:/deepfake_detection_project/data/processed_frames/test/real'
    predictor_path = 'D:/deepfake_detection_project/models/shape_predictor_68_face_landmarks.dat'
    output_dir = 'data/landmarks/video_name'
    extract_landmarks_from_frames(frame_dir, predictor_path, output_dir)

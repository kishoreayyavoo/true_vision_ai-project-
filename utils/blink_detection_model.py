import cv2
import dlib
import numpy as np

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("utils/shape_predictor_68_face_landmarks.dat")

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES = 3

def detect_blinks(video_path):
    cap = cv2.VideoCapture(video_path)
    blink_count = 0
    ear_consecutive_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0

            if ear < EYE_ASPECT_RATIO_THRESHOLD:
                ear_consecutive_frames += 1
            else:
                if ear_consecutive_frames >= EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES:
                    blink_count += 1
                ear_consecutive_frames = 0

            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_blinks('D:/deepfake_detection_project/data/raw_videos/real')

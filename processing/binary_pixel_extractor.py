import cv2
import os
import numpy as np
def binary_pixel_extraction_from_landmarks(landmark_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for landmark_file in os.listdir(landmark_dir):
        landmark_path = os.path.join(landmark_dir, landmark_file)
        img = cv2.imread(landmark_path, 0)  
        _, binary_img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
        output_binary_path = os.path.join(output_dir, f"binary_{landmark_file}")
        np.savetxt(output_binary_path.replace(".jpg", ".csv"), binary_img, fmt='%d', delimiter=",")
        cv2.imwrite(output_binary_path.replace(".csv", ".jpg"), binary_img * 255)
if __name__ == "__main__":
    landmark_dir = 'data/landmarks/video_name'
    output_dir = 'data/binary_frames/video_name'
    binary_pixel_extraction_from_landmarks(landmark_dir, output_dir)

import cv2
import os
import pandas as pd

def extract_frames(video_path, output_dir, label, csv_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(video_path)
    count = 0
    frame_labels = []

    while True:
        success, image = video.read()
        if not success:
            break
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, image)
        frame_labels.append({'filename': f"frame_{count:04d}.jpg", 'label': label})
        count += 1

    video.release()

    df = pd.DataFrame(frame_labels)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    print(f"Frames extracted to {output_dir} and labels saved to {csv_file}")

if __name__ == "__main__":
    real_video_path = 'D:/deepfake_detection_project/data/raw_videos/real/nancy-pelosi-doctored-split (1).mp4'
    fake_video_path = 'D:/deepfake_detection_project/data/raw_videos/fake'
    
    real_frames_dir = 'D:/deepfake_detection_project/data/processed_frames/train/real'
    fake_frames_dir = 'D:/deepfake_detection_project/data/processed_frames/train/fake'
    labels_csv_path = 'D:/deepfake_detection_project/data/label.csv'

    extract_frames(real_video_path, real_frames_dir, label=1, csv_file=labels_csv_path)
    extract_frames(fake_video_path, fake_frames_dir, label=0, csv_file=labels_csv_path)

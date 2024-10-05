import os
import numpy as np
def convert_frames_to_csv(frame_dir, label, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir)]
    data = []
    for frame_file in frame_files:
        try:
            for encoding in ['utf-8', 'latin1', 'ISO-8859-1']:
                try:
                    with open(frame_file, 'r', encoding=encoding) as file:
                        frame = np.genfromtxt(file, delimiter=',')
                    break 
                except UnicodeDecodeError:
                    continue  
            else:
                raise ValueError(f"Cannot read file {frame_file} with available encodings.")
        except Exception as e:
            print(f"Error processing file {frame_file}: {e}")
            continue
        frame_flat = frame.flatten()
        data.append(np.append(frame_flat, label))
    
    np.savetxt(output_csv, data, delimiter=",", fmt='%d')
if __name__ == "__main__":
    frame_dir = 'D:/deepfake_detection_project/data/binary_frames/video_name'
    label = 1  
    output_csv = 'data/frames_csv/video_name_frames.csv'
    convert_frames_to_csv(frame_dir, label, output_csv)

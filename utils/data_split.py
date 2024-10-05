import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, test_dir, train_split=0.7, val_split=0.2, test_split=0.1):
    all_files = os.listdir(source_dir)
    random.shuffle(all_files)

    train_files = all_files[:int(len(all_files) * train_split)]
    val_files = all_files[int(len(all_files) * train_split):int(len(all_files) * (train_split + val_split))]
    test_files = all_files[int(len(all_files) * (train_split + val_split)):]

    def move_files(files, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file in files:
            shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)

if __name__ == "__main__":
    split_data(
        "D:/deepfake_detection_project/data/processed_frames/train/real",
        "D:/deepfake_detection_project/data/processed_frames/train/real",
        "D:/deepfake_detection_project/data/processed_frames/val/real",
        "D:/deepfake_detection_project/data/processed_frames/test/real"
    )
    split_data(
        "D:/deepfake_detection_project/data/processed_frames/train/fake",
        "D:/deepfake_detection_project/data/processed_frames/train/fake",
        "D:/deepfake_detection_project/data/processed_frames/val/fake",
        "D:/deepfake_detection_project/data/processed_frames/test/fake"
    )

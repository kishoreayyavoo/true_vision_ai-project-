import os
import csv

def generate_labels(data_dir, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for label_dir, label in [('real', 1), ('fake', 0)]:
            for sub_dir in ['train', 'val', 'test']:
                directory = os.path.join(data_dir, sub_dir, label_dir)
                for filename in os.listdir(directory):
                    if filename.endswith(('.jpg', '.png')):
                        writer.writerow({'filename': filename, 'label': label})

if __name__ == "__main__":
    data_dir = 'D:/deepfake_detection_project/data/processed_frames'
    csv_file = 'D:/deepfake_detection_project/data/label.csv'
    generate_labels(data_dir, csv_file)

True Vision Ai

This project implements a True Vision Ai system using a CNN for Video Analysis, MTCNN for image analysis and GRU used to train the model and also implements the live extension to detect the fake content in social media.


Features 

Image:
It detects real and fake images based on facial landmarks. Utilizes MTCNN for accurate face detection and landmark extraction and it supports single image classification via command line input.

Video: 
It detects real and fake videos based on facial landmarks. Utilizes CNN for accurate face detection and landmark extraction.
the video is extracted frame by frame and converted into binary gray scale and train the model predict the result.


Live Extension:
It detects real and fake videos based on facial landmarks. when the user turn on the extension and it extract the frames from the video and converted into binary gray scale and train the model predict the result.



## Requirements

images:

- TensorFlow
- Keras
- OpenCV
- NumPy
- MTCNN

Video:
- CNN
- intel api - Openvino
- TensorFlow
- Keras
- OpenCV
- NumPy

Live Extension:
- html:
- javaScript:
- css

Run the code

image:
 give data
 and run the train_model.py and run detect_deepfake.py

 # python detect_deepfake.py

 video:

 1. Give the Real and Fake video in the Rawdata Folder.
 2. Run the Extraction.py file.
 3. Run the landmarks_Extraction.py file , conversion file.
 4. Run the train the model file trian1.py.
 5. Validate the file.
 6. run the test.py
 7. Run the prediction file.


 live Extesion :
 1.  Start the Live Extension.
 2. Run the Extraction.py file.
 3. Run the landmarks_Extraction.py file , conversion file.
 4. Run the train the model file trian1.py.
 5. Validate the file.
 6. run the test.py
 7. Run the prediction file.


 
 




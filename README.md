# CS470-Final-Project
Emotion Mate: User Emotion Reactive AI Robot UsingCNN Image Classification Model

# Team information
- Team 31
- 20170719 Han Seunghee
- 20160633 Cho Jaemin
- 20170057 Geon Gim

# File organization

- CNN Image Classification Model
  - Additional_data
    - EmoPy datset, JAFFE dataset, Korean celebrities dataset
  - CNN_result
    - Result figure of CNN model
  - Preprocess
    - Python code for preprocess to convert image into csv file
  - checkpoints
     - Checkpoint of our model
  - Kaggle_dataset
    - Basic dataset from Kaggle facial expression recognition challenge
  - Final_dataset
    - Final dataset (Kaggle + Emopy + JAFFE + Korean celebrities)
  - model_test.py
    - test code for using our CNN model on local
  - CNN_emotion_recognition.ipynb
    - Jupyter notebook file to train our model on Google Colab
  - cnn_emotion_recogntion.py
    - python code which is converted from "CNN_emotion_recognition.ipynb"

- Reactive AI Robot Software
  - Driving_Part
    - Driving_Part.ino
      - software for driving part (OpenCV 9.04 board). 
    - motions.ino
      - software for driving part (OpenCV 9.04 board). 
  - Facial_Expression_Part
    - Facial_Expression_Part.ino
      - software for facial expression part (Arduino MEGA board).

- Service Software
  - Emotion_Mate.py
    - main service code.
    - It saves webcam pictures and cropped face pictures to test_output folder. 
  - Util Software (Softwares used for environment test)  
    - face_from_image.py
      - extract face from images in image folder, save to output folder (number of image should me provided).
    - csv_to_image.py
      - make image from csv file (dataset).
    - camera.py
      - check webcam works well (streaming).
 
- Team31_Fianl_Repot.pdf
  - Final Report of our project
 
- 31.pptx
  - Presentation material of our project

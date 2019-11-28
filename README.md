# CS470-Final-Project
# Emotion Mate: User Emotion Reactive AI Robot UsingCNN Image Classification Model

# Team information
- Team 31
- 20170719 Han Seunghee
- 20160633 Cho Jaemin
- 20170057 Geon Gim

# File organization

- CNN Image Classification Model

- Reactive AI Robot Software
  - Driving Part.ino
    - software for driving part (OpenCV 9.04 board).
    
  - Facial Expression Part.ino
    - software for facial expression part (Arduino MEGA board).

- Service Software
  - Emotion_Mate.py
    - main service code.
    - It saves webcam pictures and cropped face pictures to test_output folder.
  
  - Util Software
  
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

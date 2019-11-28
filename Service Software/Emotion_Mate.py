#main service code
#implemented for project: emotion mate
import cv2
import sys
import time
import serial

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

import torchvision
import torchvision.models as models  # train or load pre-trained models
from torchvision import transforms, utils, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

#Serial connect
arduino = serial.Serial("COM8", baudrate=9600, timeout=1)
openCM = serial.Serial("COM6", baudrate=9600, timeout=1)

#class for webcam
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        return frame

#model
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv = nn.Sequential(
            #3 224 128
            nn.Conv2d(1, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #64 112 64
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 1),
            nn.MaxPool2d(2, 1),
            nn.Dropout(0.5),
            #128 56 32
            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #256 28 16
            nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(512),
        )
        #512 7 4
        self.avg_pool = nn.AvgPool2d(7)
        #512 1 1
        self.classifier = nn.Linear(512, 7)
        self.softmax = nn.Softmax()

    def forward(self, x):
        features = self.conv(x)
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x

def imshow(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    img_numpy = np.array(image, 'uint8')
    cv2.imwrite('gray_test.jpg', img_numpy)
    plt.imshow(image)
    plt.pause(0.001)

def load_our_model(device=device):
    model = VGGNet()
    optimizer = optim.Adam(model.parameters(),lr=0.008)
    ckpt_path = 'checkpoints\lastest.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    model = model.eval().to(device)
    best_acc = ckpt['best_acc']
    print('best_acc: ', best_acc)
    return model

def load_img(img_mat, imsize=48):
    preprocess = transforms.Compose([
        transforms.Resize(imsize),            # fixed size for both content and style images
        transforms.CenterCrop(imsize),        # crops the given PIL Image at the center.
        transforms.ToTensor(),                # range between 0 and 1
    ])
    img_mat_gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
    print(img_mat_gray.shape)
    #img = Image.open(filename).convert('L')
    #img_trans = np.transpose(img_mat_gray, (2, 0, 1))
    #print(img_trans[0])
    #print(img_trans[1])
    #print(img_trans[2])
    #img_trans = img_trans.squeeze(0)
    img_pil = Image.fromarray(img_mat_gray.astype('uint8'))
    img_pil.convert('L')
    img = preprocess(img_pil)
    return img

def predict_emotion(model, img, device=device):
    prediction_score = F.softmax(model(img.unsqueeze(0).to(device)), dim = 1)
    print(prediction_score)
    prediction = prediction_score.argmax().item()
    return prediction, prediction_score

#i = 0
cam = VideoCamera()
model = load_our_model()

emotion_score_arr = [-3, 0, 0, 0, 0, 0]
last_state = -1
max_num = 0
arduino.write('3'.encode('utf-8'))

while True:
    #imagePath = Path + imgname
    #print("testimg", i)
    #imagePath = Path + str(i) + ".png"
    #print("path: ", imagePath)
    
    #get image from webcam
    image = cam.get_frame()
    #cv2.imshow("image", image)
    if image is None:
        print("no image error!")
        continue
    #cv2.imshow("image", image)
    output_name1 = 'output_image\\test_output(original).png'
    #print("output: ", output_name)
    status = cv2.imwrite(output_name1, image)
    #make image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #find face part from image
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces!".format(len(faces)))

    max_size = -1
    max_img_num = -1
    for n, (x, y, w, h) in enumerate(faces):
        size = w * h
        if size > max_size:
            max_img_num = n
    #if face found
    if max_img_num != -1:
        #crop and resize face part from image
        #print(faces[max_img_num][0], faces[max_img_num][1], faces[max_img_num][2], faces[max_img_num][3])
        trim_image = image[faces[max_img_num][1]:faces[max_img_num][1]+faces[max_img_num][3], faces[max_img_num][0]:faces[max_img_num][0]+faces[max_img_num][2]]
        resize_img = cv2.resize(trim_image, dsize=(320, 320), interpolation=cv2.INTER_AREA)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #save image
        output_name = 'output_image\\test_output(webcam).png'
        #print("output: ", output_name)
        status = cv2.imwrite(output_name, resize_img)
        
        #print(trim_image.shape)
        #classify image with trained model
        model_img = load_img(trim_image) 
        emtion_classify, emotion_score = predict_emotion(model=model, img=model_img)
        #emotion_score = emotion_score
        score_value = emotion_score[0].detach().numpy()
        #print("score value:", score_value)

        #update score of each emotion class
        if (emtion_classify == 0) or (emtion_classify == 1) or (emtion_classify == 2) or (emtion_classify == 3) or (emtion_classify == 4) or (emtion_classify == 5):
            for i in range(6):
                #print("for: ", score_value[i])
                if score_value[i] < 0.12:
                    emotion_score_arr[i] -= 3
                elif score_value[i] < 0.13:
                    emotion_score_arr[i] -= 1
                elif score_value[i] < 0.2:
                    emotion_score_arr[i]
                elif score_value[i] < 0.25:
                    #print(i, "add")
                    emotion_score_arr[i] += 1
                elif score_value[i] < 1:
                    #print(i, "add")
                    emotion_score_arr[i] += 4
                else:
                    print("score value error")
                    break
            emotion_score_arr[emtion_classify] += 1
            #print(emtion_classify, "add")
        elif emtion_classify == 6: # neutral
            emotion_score_arr[0] -= 1
            emotion_score_arr[1] -= 1
            emotion_score_arr[2] -= 1
            emotion_score_arr[3] -= 1
            emotion_score_arr[4] -= 1
            emotion_score_arr[5] -= 1
        else:
            print("emotion classify error")
            break    

        print("emotion:", emotion[emtion_classify])
        print("score: ", emotion_score_arr)

        #trigger to robot
        max_num = 0
        max_emotion = -1
        max_val = 0
        max_arr = []
        for i in range(6):
            if emotion_score_arr[i] >= 12:
                max_num += 1
                max_arr.append(i)
        
        for i in max_arr:
            if emotion_score_arr[i] > max_val:
                max_emotion = i
                max_val = emotion_score_arr[i]
        #print("max arr: ", max_arr , "max_emotion: ", max_emotion)
        
        max_emotion = str(max_emotion)
        #print(max_emotion)
        if max_num == 0:
            print("no emotion recognized")
        elif max_num == 1:
            emotion_score_arr[0] = 0
            emotion_score_arr[1] = 0
            emotion_score_arr[2] = 0
            emotion_score_arr[3] = 0
            emotion_score_arr[4] = 0
            emotion_score_arr[5] = 0
            arduino.write(max_emotion.encode('utf-8'))
            openCM.write(max_emotion.encode('utf-8'))
            time.sleep(10)

        elif max_num == 2:
            emotion_score_arr[0] = 0
            emotion_score_arr[1] = 0
            emotion_score_arr[2] = 0
            emotion_score_arr[3] = 0
            emotion_score_arr[4] = 0
            emotion_score_arr[5] = 0
            arduino.write(max_emotion.encode('utf-8'))
            openCM.write(max_emotion.encode('utf-8'))
            time.sleep(10)

        elif max_num == 3:
            emotion_score_arr[0] = 0
            emotion_score_arr[1] = 0
            emotion_score_arr[2] = 0
            emotion_score_arr[3] = 0
            emotion_score_arr[4] = 0
            emotion_score_arr[5] = 0
            arduino.write(max_emotion.encode('utf-8'))
            openCM.write(max_emotion.encode('utf-8'))
            time.sleep(10)

        else:
            print("number error")
            break
        #print('Arduino reply: ', arduino.read(), 'openCM reply: ', openCM.read())
        
        for i in range(6):
            if emotion_score_arr[i] < -5:
                emotion_score_arr[i] = -5
            if emotion_score_arr[i] > 12:
                emotion_score_arr[i] = 12

    else:
        print("no face founded")
    
    time.sleep(1)

#emotion code
# 0: 'Angry'
# 1: 'Disgust'
# 2: 'Fear'
# 3: 'Happy'
# 4: 'Sad'
# 5: 'Surprise'
# 6: 'Neutral'




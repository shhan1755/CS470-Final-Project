import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform

import torchvision
import torchvision.models as models  # train or load pre-trained models
from torchvision import transforms, utils, datasets

#torch.manual_seed(470)
#torch.cuda.manual_seed(470)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

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
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    model = model.eval().to(device)
    best_acc = ckpt['best_acc']
    print('best_acc: ', best_acc)
    return model


def load_img(filename, imsize=48):
    preprocess = transforms.Compose([
        transforms.Resize(imsize),            # fixed size for both content and style images
        transforms.CenterCrop(imsize),        # crops the given PIL Image at the center.
        transforms.ToTensor(),                # range between 0 and 1
    ])
    img = Image.open(filename).convert('L')
    img = preprocess(img)
    return img

def predict_emotion(model, img, device=device):
    prediction = F.softmax(model(img.unsqueeze(0).to(device))).argmax().item()
    return prediction

model = load_our_model()
test0 = load_img(filename='test0.PNG')
test1 = load_img(filename='test1.jpg')
test2 = load_img(filename='test2.jpg')
test3 = load_img(filename='test3.png')
test4 = load_img(filename='test4.png')

imshow(test0)
imshow(test1)
'''
imshow(test2)
imshow(test3)
imshow(test4)
'''


test_prediction0 = predict_emotion(model=model, img=test0)
test_prediction1 = predict_emotion(model=model, img=test1)
test_prediction2 = predict_emotion(model=model, img=test2)
test_prediction3 = predict_emotion(model=model, img=test3)
test_prediction4 = predict_emotion(model=model, img=test4)

print(emotion[test_prediction0])
print(emotion[test_prediction1])
print(emotion[test_prediction2])
print(emotion[test_prediction3])
print(emotion[test_prediction4])



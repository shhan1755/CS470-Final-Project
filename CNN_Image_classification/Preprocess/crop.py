import os
import sys
import random
import numpy as np
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(48),  
    transforms.CenterCrop(48),    
])
'''
Input Example : python crop.py /path/to/folder
'''

### ---- Step 0 - Argument Check ---- ###
if len(sys.argv) < 1:
    print('Need path data')

### ---- Step 1 - Load the path ---- ###
path = sys.argv[1] # path containing fine tuning image data
folders = os.listdir(path)

### ---- Step 2 - Construct Map direct emotion and label ---- ###
emotion = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotion_csv = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
column = 'emotion,pixels,Usage\n'
whole_datas = 0
whole_lines = []

print('The number of classification', len(folders))
print('Contents of folders: ', folders)

### ---- Step 3 - Image to csv file format ---- ###
for i in range(len(folders)):
    emotion_path = os.listdir(path+'/'+folders[i])
    print(emotion_path)
    whole_datas += len(emotion_path)
    for j in range(len(emotion_path)):
        img = Image.open(path+'/'+folders[i]+'/'+emotion_path[j]).convert('L') # Load image to gray scale PIL image
        img = preprocess(img) # Crop & Resize
        img_np = np.array(img) # PIL image to numpy array
        img_list = np.reshape(img_np, (1, 48*48)).squeeze(0).tolist() # Numpy array to 1-dimensional list
        img_list = list(map(str, img_list)) # int list to str list
        img_str = ' '.join(img_list) # list to str
        emotion_csv[emotion[folders[i]]].append('%d,%s,Training\n' %(emotion[folders[i]], img_str)) # append each line of labeled image and label

print('The number of whole datas:', whole_datas)        

### ---- Step 4 - Write data to csv file ---- ###
csv_file = open('./fine_train.csv', 'w')
csv_file.write(column) 

for em in emotion_csv:
    whole_lines += emotion_csv[em]

random.shuffle(whole_lines)

for line in whole_lines:
    csv_file.write(line)
    

csv_file.close()

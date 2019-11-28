from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils, datasets


filename = '2_0.jpg'

img = Image.open(filename).convert('L')

img_array = np.array(img)

preprocess = transforms.Compose([
    transforms.Resize(48),            # fixed size for both content and style images
    transforms.CenterCrop(48),        # crops the given PIL Image at the center.
    transforms.ToTensor(),                # range between 0 and 1
])

#img_tensor = torch.as_tensor(img_array)
img_tensor = preprocess(img)



print(img_array)
print(img_tensor)
count = 0
for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        if img_array[i][j] > 200:
            count += 1

print(count)

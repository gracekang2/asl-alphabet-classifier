from cv2 import cv2
# import torch
# import torchvision
# from torchvision import datasets, transforms
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# import numpy as np
# pylint: disable=unused-wildcard-import
from fastai.vision.all import *

# workaround for running on Windows machine
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# class NeuralNet2(nn.Module):
#     def __init__(self):
#         super(NeuralNet2, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,5))
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.dropout3 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(12544, 256)
#         self.fc2 = nn.Linear(256, 24)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=(2,2))
#         x = self.dropout1(x)

#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, kernel_size=(2,2), stride=(2,2))
#         x = self.dropout2(x)
        
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout3(x)
        
#         x = self.fc2(x)

#         return x
# function used to label data when training model
def label_func(f):
    return f[0]

# model = torch.load("./models/custom_rms.pkl")
learn = load_learner(fname="./models/model_aug_128.pkl")
FRAME_SIZE = 128

def preprocess(img):
    # img_transforms = transforms.Compose([
    #     transforms.Resize(64),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     transforms.Grayscale(num_output_channels=1),
    # ])
    # return torch.reshape(img_transforms(img), (1, 1, 64, 64))

    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return src_gray

def predict(img):
    # classes = {n: chr(n + 65) if n < 9 else chr(n + 1 + 65) for n in range(24)}

    # outputs = model(preprocess(img))
    # _, predicted = torch.max(outputs.data, 1)
    # letter = predicted[0].item()
    # return classes.get(letter, letter)

    letter,_,probs = learn.predict(preprocess(img))
    return letter, probs

def translate():
    c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    indices = {n if n < 9 else n + 1 : n for n in range(24)} # alphabet index --> model index 

    x = 100
    y = 100
    dim = 256

    while c.isOpened():
        ret, frame = c.read()

        cv2.rectangle(frame, (x,y),(x+dim, y+dim),(255,0,0),2)
        # box = Image.fromarray(frame[x:x+dim, y:y+dim])
        # letter = predict(box)
        box = frame[x:x+dim, y:y+dim]
        letter, p = predict(box)

        if letter.isalpha() and ord(letter) - 97 in indices:
            prob = int(p[indices[ord(letter) - 97]].item() * 100)

            if prob >= 70:
                cv2.putText(frame, "%s: %d%%" % (letter, prob), (int(x), y+dim+50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA) 
            
            elif prob >= 40:
                cv2.putText(frame, "maybe %s: %d%%" % (letter, prob), (int(x), y+dim+50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA) 


        cv2.imshow('frame', frame)            

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    c.release()
    cv2.destroyAllWindows()

translate()

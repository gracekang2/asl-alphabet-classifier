from cv2 import cv2
# pylint: disable=unused-wildcard-import
from fastai.vision.all import *

# workaround for running on Windows machine
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# function used to label data when training model
def label_func(f):
    return f[0]

learn = load_learner(fname="./export.pkl")

def translate():
    c = cv2.VideoCapture(0)

    x = 100
    y = 100
    dim = 256

    while True:
        ret, frame = c.read()

        cv2.rectangle(frame, (x,y),(x+dim, y+dim),(255,0,0),2)
        box = frame[x:x+dim, y:y+dim]
        letter,_,probs = learn.predict(box)

        cv2.putText(frame, letter, (int(x+dim/2), y+50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA) 

        cv2.imshow('frame', frame)            

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    c.release()
    cv2.destroyAllWindows()

translate()

# test_img = cv2.imread("testC.jpg")
# letter,_,probs = learn.predict(test_img)
# print(letter)

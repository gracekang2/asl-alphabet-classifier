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

learn = load_learner(fname="./models/model_aug_128.pkl")
FRAME_SIZE = 128

def preprocess(img):
    img = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return src_gray

def predict(img):
    letter,_,probs = learn.predict(preprocess(img))
    return letter, probs

def translate():
    c = cv2.VideoCapture(0)
    black_img = np.zeros((250, 500, 3), np.uint8)
    cv2.imshow('text', black_img)

    x = 100
    y = 100
    dim = 256

    while c.isOpened():
        ret, frame = c.read()

        cv2.rectangle(frame, (x,y),(x+dim, y+dim),(255,0,0),2)
        box = frame[x:x+dim, y:y+dim]
        letter, p = predict(box)

        if letter.isalpha():
            sub = 97
            if ord(letter) >= 106:
                sub += 1
            
            prob = p[ord(letter) - sub]
            if prob * 100 >= 80:
                cv2.putText(frame, letter + ": " + str(prob), (int(x+dim/2), y+dim+50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA) 

        cv2.imshow('frame', frame)            

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    c.release()
    cv2.destroyAllWindows()

translate()

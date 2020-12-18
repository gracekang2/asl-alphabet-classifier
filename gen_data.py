from cv2 import cv2
import numpy as np

"""
    press key of letter that is being photographed
    e.g. w if ASL sign is w
    will print DONE after COUNT photos surpassed
"""

COUNT = 250
x = 100
y = 100
dim = 224

def process(src):
    src = src[x:x+dim, y:y+dim]	
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', src_gray)
    return src_gray
    

def capture():
    c = cv2.VideoCapture(0)
    counter = 0

    while c.isOpened():
        ret, frame = c.read()

        cv2.rectangle(frame, (x,y),(x+dim, y+dim),(255,0,0),2)
        cv2.imshow('frame', frame)            

        key = cv2.waitKey(1)
        if key == ord('1'):
            break
        elif key >= 10 and key <= 132 and chr(key).isalpha():
            letter = chr(key)
            img = process(frame)
            fname = "./images/"+letter+str(counter)+".jpg"
            counter += 1
            cv2.imwrite(fname, img)

            if counter >= COUNT:
                print("DONE")

    c.release()
    cv2.destroyAllWindows()

capture()
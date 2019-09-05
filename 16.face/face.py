#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
"""
@project: face
@file: face.py
@version: 1.0
@IDE: PyCharm
@author: gyh
@contact: gyhnice@163.com
@time: 19-8-27 下午5:47
@desc: None
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_detect():
    face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
    img = cv2.imread("./Andrew Ng.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,2)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("detect",img)

def video_detect():
    face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")

    capture=cv2.VideoCapture("./demo_video.mp4")
    while True:
        ret,frame = capture.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        eyes = eye_cascade.detectMultiScale(gray,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        cv2.imshow("detect",frame)
        if cv2.waitKey(25) & 0xff == ord("q"):
            break



cv2.namedWindow("detect")


# img_detect()
video_detect()

cv2.waitKey(0)
cv2.destroyAllWindows()
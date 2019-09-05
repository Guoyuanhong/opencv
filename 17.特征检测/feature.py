#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
"""
@project: 17.特征检测
@file: feature.py
@version: 1.0
@IDE: PyCharm
@author: gyh
@contact: gyhnice@163.com
@time: 19-8-30 下午3:58
@desc: None
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris():
    img=cv2.imread("./1.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,23,0.04)
    # 角点检测
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow("harris",img)



harris()
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2 as cv
import numpy as np

def color_convers(image):   #不同色彩空间转换
    grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY) #rgb->grey
    cv.imshow("grey",grey)
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV) #rgb->hsv
    cv.imshow("hsv",hsv)

rsc = cv.imread("/home/gyh/opencv/lena.jpg")
color_convers(rsc)
cv.waitKey(0)
cv.destroyAllWindows()
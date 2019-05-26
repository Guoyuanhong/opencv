import cv2 as cv
import numpy as np

#图像分割，合并
rsc = cv.imread("/home/gyh/opencv/lena.jpg")

b,g,r= cv.split(rsc)    #利用split方法分割
cv.imshow("blue" ,b)
cv.imshow("green" ,g)
cv.imshow("red" ,r)

#通道合并

img = cv.merge([b,g,r])
cv.imshow("img",img)

#改变某个通道的值
rsc[:,:,2]=0
cv.imshow("rsc",rsc)
cv.waitKey(0)
cv.destroyAllWindows()
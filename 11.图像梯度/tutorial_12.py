import cv2 as cv
import numpy as np

src = cv.imread("/home/gyh/opencv/lena.jpg")

"""
梯度简单来说就是求导。
OpenCV 提供了三种不同的梯度滤波器,或者说高通滤波器:Sobel,Scharr 和 Laplacian
"""
#laplacian
laplace = cv.Laplacian(src,-1)
sobel = cv.Sobel(src,-1,0,1)#01表示对x求梯度，10表示对y

cv.imshow("src",src)
cv.imshow("IMG",sobel)
cv.waitKey(0)
cv.destroyAllWindows()

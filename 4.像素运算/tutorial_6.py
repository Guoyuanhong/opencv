import cv2 as cv
import numpy as np


#像素的加减乘除，均值，标准差运算

def demo(img1,img2):
    #相加
    m1=cv.add(img1,img2)
    cv.imshow("add",m1)
    #相减
    m2 = cv.subtract(img1,img2)
    cv.imshow("sub", m2)
    #相除
    m3 = cv.divide(img1,img2)
    cv.imshow("div",m3)
    #相乘
    m4=cv.multiply(img1,img2)
    cv.imshow("mul",m4)
    #均值
    m5=cv.mean(img1)
    cv.imshow("mean",m5)
    #标准差
    M1,dev1 = cv.meanStdDev(img1)
    M2,dev2 = cv.meanStdDev(img2)
    print(M1)
    print(M2)
    print(dev1)
    print(dev2)

src1 = cv.imread("1.jpg")
src2 = cv.imread("2.jpg")

demo(src1,src2)

cv.waitKey(0)
cv.destroyAllWindows()

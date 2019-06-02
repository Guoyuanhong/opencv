import cv2 as cv
import numpy as np

src=cv.imread("/home/gyh/opencv/lena.jpg")
img = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#全局阈值
ret,thresh1=cv.threshold(img,120,255,cv.THRESH_BINARY)
"""
像素值高于阈值时，给这个像素赋予一个新值（可能是白色），否则我们给它赋予另外一种颜色（也许是黑色）。
这个函数就是 cv2.threshhold()。这个函数的第一个参数就是原图像，原图像应该是灰度图。
第二个参数就是用来对像素值进行分类的阈值。
第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值。 
OpenCV提供了多种不同的阈值方法，这是有第四个参数来决定的。这些方法包括：
cv2.THRESH_BINARY
cv2.THRESH_BINARY_INV
cv2.THRESH_TRUNC
cv2.THRESH_TOZERO
cv2.THRESH_TOZERO_INV
"""

#局部阈值
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
"""
此时的阈值是根据图像上的
每一个小区域计算与其对应的阈值。因此在同一幅图像上的不同区域采用的是
不同的阈值,从而使我们能在亮度不同的情况下得到更好的结果。
这种方法需要我们指定三个参数,返回值只有一个。
• Adaptive Method- 指定计算阈值的方法。
– cv2.ADPTIVE_THRESH_MEAN_C:阈值取自相邻区域的平
均值
– cv2.ADPTIVE_THRESH_GAUSSIAN_C:阈值取值相邻区域
的加权和,权重为一个高斯窗口。
• Block Size - 邻域大小(用来计算阈值的区域大小)。
• C - 这就是是一个常数,阈值就等于的平均值或者加权平均值减去这个常
数
"""
#OTSU阈值

ret1,th4 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
"""
这里用到到的函数还是 cv2.threshold(),但是需要多传入一个参数
(flag):cv2.THRESH_OTSU。这时要把阈值设为 0。然后算法会找到最
优阈值,这个最优阈值就是返回值 retVal。如果不使用 Otsu 二值化,返回的
retVal 值与设定的阈值相等。
"""
print(ret1)
cv.imshow("src",src)
cv.imshow("threach1",thresh1)
cv.imshow("threach2",th2)
cv.imshow("threach3",th3)
cv.imshow("threach4",th4)
cv.waitKey(0)
cv.destroyAllWindows()
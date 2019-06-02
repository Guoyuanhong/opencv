import cv2 as cv
import numpy as np

src = cv.imread("/home/gyh/opencv/lena.jpg")
img = cv.Canny(src,100,150)
"""
cv2.Canny(),就可以完成以上几步。
让我们看如何使用这个函数。这个函数的第一个参数是输入图像。第二和第三
个分别是 minVal 和 maxVal。第三个参数设置用来计算图像梯度的 Sobel
卷积核的大小,默认值为 3。最后一个参数是 L2gradient,它可以用来设定
求梯度大小的方程。如果设为 True,就会使用我们上面提到过的方程,否则
使用方程:Edge − Gradient (G) = |G 2 x | + |G 2 y | 代替,默认值为 False。
"""

cv.imshow("src",src)
cv.imshow("img",img)
cv.waitKey(0)
cv.destroyAllWindows()

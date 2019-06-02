import cv2 as cv
import numpy as np

src = cv.imread("1.png")
kenerl = np.ones((5,5),np.uint8)
#腐蚀
img1=cv.erode(src,kenerl)
"""
一般情况下对二值化图像进行的操作,这个操作会把前景物体的边界腐蚀掉(但是前景仍然是白色)
卷积核沿着图像滑动,如果与卷积核对应的原图像的所有像素值都是 1,那么中心元素就保持原来的像素值,否则就变为零。
"""
#膨胀
img2 = cv.dilate(src,kenerl)
"""
与腐蚀相反,与卷积核对应的原图像的像素值中只要有一个是 1,中心元素的像素值就是 1。
所以这个操作会增加图像中的白色区域(前景)。
一般在去噪声时先用腐蚀再用膨胀。因为腐蚀在去掉白噪声的同时,也会使前景对象变小。
所以我们再对他进行膨胀。这时噪声已经被去除了,不会再回来了,但是前景还在并会增加。
膨胀也可以用来连接两个分开的物体。
"""
#开运算，先腐蚀在膨胀
img3 = cv.morphologyEx(src,cv.MORPH_OPEN,kenerl)
#闭运算，先膨胀，再腐蚀
img4 = cv.morphologyEx(src,cv.MORPH_CLOSE,kenerl)
#形态学梯度
img5 = cv.morphologyEx(src,cv.MORPH_GRADIENT,kenerl)

cv.imshow("src",src)
cv.imshow("img",img5)
cv.waitKey(0)
cv.destroyAllWindows()
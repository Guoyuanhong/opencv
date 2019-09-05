#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
"""
@project: Review
@file: review1.py
@version: 1.0
@IDE: PyCharm
@author: gyh
@contact: gyhnice@163.com
@time: 19-8-26 下午1:01
@desc: None
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def colorspace():
    """
    色彩空间转换以及颜色跟踪
    """
    src=cv2.imread("./1.png",cv2.IMREAD_COLOR)
    hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)

    lower_blue = np.array([96, 210, 174])
    upper_blue = np.array([114, 255, 216])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow("mask",mask)

def threshold():
    """
    全局阈值，自适应阈值和Otsu’s 二值化
    """
# 全局阈值
    src=cv2.imread("./2.png")
    #cv2.imshow("2.png",src)

    #高于阈值置255，低于置0
    ret,thres1=cv2.threshold(src,127,255,cv2.THRESH_BINARY)
    #cv2.imshow("BINARY",thres1)
    # 高于阈值置0，低于置255
    ret, thres2 = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("BINARY_INV", thres2)
    #高于阈值置成阈值，低于阈值不变
    ret, thres3 = cv2.threshold(src, 127, 255, cv2.THRESH_TRUNC)
    #cv2.imshow("TRUNC", thres3)
    #低于阈值置0，高于不变
    ret, thres4 = cv2.threshold(src, 127, 255, cv2.THRESH_TOZERO)
    # cv2.imshow("TOZERO", thres4)
    # 低于阈值不变，高于置0
    ret, thres5 = cv2.threshold(src, 127, 255, cv2.THRESH_TOZERO_INV)
    #cv2.imshow("TOZERO_INV", thres5)

# 自适应阈值
    img=cv2.imread("./3.png")
    #中值滤波，去椒盐噪声
    img = cv2.medianBlur(img,5)
    #cv2.imshow("3.png", img)

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #cv2.imshow("THRESH_BINARY", th1)8

    #blocksize=11,c=2,这个函数要单通道图片
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #cv2.imshow("ADAPTIVE_THRESH_MEAN_C", th3)

# Otsu’s 二值化
    #简单来说就是对一副双峰图像自动根据其直方图计算出一个阈值。
    #(对于非双峰图像,这种方法得到的结果可能会不理想)。
    img1 = cv2.imread("./4.png")
    # cv2.imshow("4.png",img1)

    #有噪声，先高斯模糊
    img1=cv2.GaussianBlur(img1,(5,5),0)

    # OTSU,要8bit，sigle-channel
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    ret,th=cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("THRESH_OTSU",th)

def filter():
    """
    filter2D,各种滤波
    LPF 帮助我们去除噪音,模糊图像。HPF 帮助我们找到图像的边缘
    """
# 自建内核滤波，2D卷积
    src = cv2.imread("./5.png")
    #cv2.imshow("5.png",src)
    # 构建5x5的平均滤波器核
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(src,-1,kernel)
    # when ddepth = -1, the output image will have the same depth as the source.
    #cv2.imshow("filter",dst)

# 图像模糊（平滑）
    #平均滤波
    # 这是由一个归一化卷积框完成的。他只是用卷积框覆盖区域所有像素的平均值来代替中心元素。如让面的卷积核
    blur = cv2.blur(src,(5,5))

    #高斯滤波
    # 现在把卷积核换成高斯核(简单来说,方框不变,将原来每个方框的值是相等的,
    # 现在里面的值是符合高斯分布的,方框中心的值最大,其余方框根据距离中心元素的距离递减,构成一个高斯小山包。
    # 原来的求平均数现在变成求加权平均数,全就是方框里的值)
    # 高斯滤波可以有效的从图像中去除高斯噪音。不考虑边界，边界也会被模糊
    blur=cv2.GaussianBlur(src,(5,5),0)
    # 需要指定高斯核的宽和高(必须是奇数)
    # 第三个参数sigmaX，sigmaY和sigmaX相等，如果两者都等于0，则它们是从ksize.width和ksize.height计算的

    # 中值滤波
    # 顾名思义就是用与卷积框对应像素的中值来替代中心像素的值。这个滤波器经常用来去除椒盐噪声。
    # 前面的滤波器都是用计算得到的一个新值来取代中心像素的值,而中值滤波是用中心像素周围(也可以使他本身)的值来取代他。
    # 他能有效的去除噪声。卷积核的大小也应该是一个奇数。
    src=cv2.imread("./7.jpg")
    # cv2.imshow("6.png",src)
    blur = cv2.medianBlur(src,9)
    #  it must be odd（奇数） and greater than 1（比1大）for example: 3, 5, 7

    # 双边滤波
    # 函数 cv2.bilateralFilter() 能在保持边界清晰的情况下有效的去除噪音。但是这种操作与其他滤波器相比会比较慢。
    # 双边滤波在同时使用空间高斯权重和灰度值相似性高斯权重。空间高斯函数确保只有邻近区域的像素对中心点有影响,
    # 灰度值相似性高斯函数确保只有与中心像素灰度值相近的才会被用来做模糊运算。
    # 所以这种方法会确保边界不会被模糊掉,因为边界处的灰度值变化比较大。
    src = cv2.imread("./9.png")
    # cv2.imshow("9.png",src)
    blur=cv2.bilateralFilter(src,9,75,75)
    # d过滤期间使用的每个像素邻域的直径。如果是非正的，它是从sigmaSpace计算出来的
    # sigmaColor，灰度值相似性高斯函数标准差
    # sigmaSpace，空间高斯函数标准差
    # cv2.imshow("blur",blur)

def Morphological():
    """
    形态学操作，腐蚀，膨胀，开运算，闭运算，形态学梯度，礼帽，黑帽
    关系：
        开运算=先腐蚀在膨胀
        闭运算=先膨胀在腐蚀
        形态学梯度=膨胀-腐蚀
        礼帽=原图-开运算
        黑帽=闭运算-原图
    """
    src = cv2.imread("./10.png")
    # cv2.imshow("10.png",src)

    # 腐蚀
    # 卷积核沿着图像滑动, 如果与卷积核对应的原图像的所有像素值都是1, 那么中心元素就保持原来的像素值, 否则就变为零。(与？)
    # 根据卷积核的大小靠近前景的所有像素都会被腐蚀掉(变为 0),所以前景物体会变小,整幅图像的白色区域会减少。
    # 这对于去除白噪声很有用,也可以用来断开两个连在一块的物体等。
    kernel = np.ones((5,5),np.uint8)
    img = cv2.erode(src,kernel,iterations=1)
    # iterations number of times erosion is applied.

    # 膨胀
    # 与腐蚀相反,与卷积核对应的原图像的像素值中只要有一个是 1,中心元素的像素值就是 1。
    # 这个操作会增加图像中的白色区域(前景)。一般在去噪声时先用腐蚀再用膨胀。
    # 因为腐蚀在去掉白噪声的同时,也会使前景对象变小。所以我们再对他进行膨胀。
    # 这时噪声已经被去除了,不会再回来了,但是前景还在并会增加。膨胀也可以用来连接两个分开的物体。
    img = cv2.dilate(src,kernel,iterations=1)

    # 开运算
    # 先进行腐蚀再进行膨胀就叫做开运算。就像我们上面介绍的那样,它被用来去除噪声。去除背景黑点
    src = cv2.imread("./11.png")
    # cv2.imshow("11.png", src)
    img = cv2.morphologyEx(src,cv2.MORPH_OPEN,kernel)

    # 闭运算
    # 先膨胀再腐蚀。它经常被用来填充前景物体中的小洞,或者前景物体上的小黑点。
    src= cv2.imread("./12.png")
    # cv2.imshow("12.png", src)
    img = cv2.morphologyEx(src,cv2.MORPH_CLOSE,kernel)


    # 形态学梯度
    # 其实就是一幅图像膨胀与腐蚀的差别。结果看上去就像前景物体的轮廓。
    src = cv2.imread("./10.png")
    img = cv2.morphologyEx(src,cv2.MORPH_GRADIENT,kernel)

    # 礼帽
    kernel = np.ones((13, 13), np.uint8)
    img = cv2.morphologyEx(src,cv2.MORPH_TOPHAT,kernel)

    # 黑帽
    img = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    # cv2.imshow("Morphological",img)

def img_gradient():
    """
    图像梯度，cv2.Sobel(),cv2.Scharr(),cv2.Laplacian()
    """
    src = cv2.imread("./3.png")
    # src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    # cv2.CV_64F 输出图像的深度(数据类型),可以使用 -1, 与原图像保持一致 np.uint8
    laplacian = cv2.Laplacian(src, -1)
    sobel = cv2.Sobel(src, -1, 0,1)  # 01表示对x求梯度，10表示对y
    schar = cv2.Scharr(src,-1,1,0)  # 01表示对x求梯度，10表示对y
    # cv2.imshow("laplacian",schar)

def canny():
    src=cv2.imread("./13.png")
    edges=cv2.Canny(src,100,200)

    plt.subplot(121),plt.imshow(src,cmap="gray")
    plt.title('Original Image'),plt.xticks([]),plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

def profile():
    """
    为了更加准确,要使用二值化图像。在寻找轮廓之前,要进行阈值化处理或者 Canny 边界检测。
    查找轮廓的函数会修改原始图像。如果你在找到轮廓之后还想使用原始图像的话,你应该将原始图像存储到其他变量中。
    在 OpenCV 中,查找轮廓就像在黑色背景中超白色物体。你应该记住,要找的物体应该是白色而背景应该是黑色。
    """
    img = cv2.imread("./14.png")
    src=np.copy(img)
    dst = cv2.GaussianBlur(src,(3,3),0)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i,contou in enumerate(contours):
        cv2.drawContours(src,contours,i,(0,0,225),-1)
    # cv2.imshow("profile",src)

def hist():
    #直方图统计
    img=cv2.imread("./1.png")
    color=("b","g","r")
    mask = np.zeros(img.shape[:2],np.uint8)
    mask[100:300,100:400]=255
    for i,col in enumerate(color):
        hists=cv2.calcHist([img],[i],mask,[256],[0,256])

        plt.plot(hists,color=col)
        plt.xlim([0,256])
    # plt.show()

    # 直方图均衡化
    # global
    img = cv2.imread("./15.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", img)
    src = cv2.equalizeHist(img)
    # cv2.imshow("hist",src)

    # CLAHE 有限对比适应性直方图均衡化(自适应的直方图均衡化)
    img = cv2.imread("./16.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("16.png", img)
    # 创建clahe的object， #clipLimit是对比度的大小，tileGridSize是每次处理块的大小（默认（8,8））
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # 应用到img
    dst=clahe.apply(np.copy(img))
    # cv2.imshow("clahe",dst)

    # 2D直方图
    img = cv2.imread("./lena.jpg")
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    src = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    plt.imshow(src,interpolation='nearest')
    # plt.show()

    # 直方图反向投影，它可以用来做图像分割,或者在图像中找寻我们感兴趣的部分

def hough_line_circle():
    # for lines
    img = cv2.imread("./3.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,100,apertureSize=3)
    # cv2.imshow("edges",edges)
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    for rho,theta in lines[30]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cv2.imshow("lines",img)

    # for circles
    img = cv2.imread("./17.png")
    # cv2.imshow("17.png",img)

    img = cv2.pyrMeanShiftFiltering(img, 10, 100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
    # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 3)

    # cv2.imshow('detected circles', img)

def watershed():
    img = cv2.imread("./17.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret,sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret,makers = cv2.connectedComponents(sure_fg)
    makers = makers+1
    makers[unknown==255]=0
    makers = cv2.watershed(img,makers)
    img[makers == -1]=[255,0,0]
    plt.imshow(img)
    plt.show()


# colorspace()
# threshold()
# filter()
# Morphological()
# img_gradient()
# canny()
# profile()
# hist()
# hough_line_circle()
# watershed()


cv2.waitKey(0)
cv2.destroyAllWindows()

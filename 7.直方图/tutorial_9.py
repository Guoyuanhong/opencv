import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#直方图histogram
def hist_img(img):
    color = ["b","g","r"]
    for i,col in enumerate(color):
        hist=cv.calcHist([img],[i],None,[256],[0,256])  #返回直方图数据
        """
        image输入图像，传入时应该用中括号[]
        括起来
        channels:：传入图像的通道，如果是灰度图像，那就不用说了，只有一个通道，值为0，如果是彩色图像（有3个通道），那么值为0, 1, 2, 中选择一个，对应着BGR各个通道。这个值也得用[]
        传入。
        mask：掩膜图像。如果统计整幅图，那么为none。主要是如果要统计部分图的直方图，就得构造相应的炎掩膜来计算。
        histSize：灰度级的个数，需要中括号，比如[256]
        ranges: 像素值的范围，通常[0, 256]，有的图像如果不是0 - 256，比如说你来回各种变换导致像素值负值、很大，则需要调整后才可以。
        """
        plt.plot(hist,color=col)    #将hist的值表示成直方图，color是这个方法的一个设置线条颜色的参数，b,g.r
        plt.xlim([0,256])   #设定坐标范围
    plt.show()  #可视化

#直方图均衡化，也就是提高对比度
def equil_hist(image):
    grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow("old",grey)
    #全局提高对比度
    cv.equalizeHist(grey)   #只能处理灰度图片
    cv.imshow("new",grey)

    #自定义对比度限制
    clahe= cv.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    #clipLimit是对比度的大小，tileGridSize是每次处理块的大小
    dst = clahe.apply(grey)
    cv.imshow("dst",dst)

#直方图比较

#直方图反向投影

#模板匹配



src = cv.imread("/home/gyh/opencv/lena.jpg")
cv.imshow("src",src)
equil_hist(src)
cv.waitKey(0)
cv.destroyAllWindows()
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

#逻辑运算
def luoji(src1,src2):
    #and
    img1 = cv.bitwise_and(src1,src2)
    cv.imshow("and",img1)
    #or
    img2 = cv.bitwise_or(src1,src2)
    cv.imshow("or",img2)
    #not
    src = cv.imread("/home/gyh/opencv/lena.jpg")
    img3 = cv.bitwise_not(src)
    cv.imshow("not",img3)
    #xor异或
    img4 = cv.bitwise_xor(src1,src2)
    cv.imshow("xor",img4)

#粗略调整对比度和亮度
def contrast_brightness_image(src , a,g):
    h,w,ch = src.shape  #获取高，宽，通道
    img = np.zeros([h,w,ch],src.dtype)
    #新建全零图片数组img,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    dst = cv.addWeighted(src,a,img,1-a,g)
    #官方：计算两个图像阵列的加权和 我的理解是按照所占比例合成两张图片。
    #addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1);
    #一共有七个参数：前4个是两张要合成的图片及它们所占比例，第5个double gamma起微调作用，第6个OutputArray dst是合成后的图片，第七个输出的图片的类型（可选参数，默认-1）
    #有公式得出两个图片加成输出的图片为：dst=src1*alpha+src2*beta+gamma

    cv.imshow("dst",dst)

src1 = cv.imread("1.jpg")
src2 = cv.imread("2.jpg")
src = cv.imread("/home/gyh/opencv/lena.jpg")

#demo(src1,src2)
#luoji(src1,src2)
contrast_brightness_image(src,0.2,10)
cv.waitKey(0)
cv.destroyAllWindows()

import cv2 as cv
import numpy as np

src = cv.imread("/home/gyh/opencv/lena.jpg")
cv.imshow("src",src)

img1 = cv.pyrDown(src)
cv.imshow("img1",img1)
#(尺寸变小,分辨率降低,信息丢失)。

img2 = cv.pyrUp(img1)
cv.imshow("img2",img2)
#(尺寸变大,但分辨率不会增加)，经过pyrDown之后再用pyrUp,不会增加分辨率。


#拉普拉斯金字塔可以有高斯金字塔计算得来,公式如下:
#L i = G i − pyrUp (G i+1 )

#高斯金字塔
def demo(src1):
    level=3
    temp = src1.copy()
    G=[]
    for i in range(level):
        dst = cv.pyrDown(temp)
        G.append(dst)
        cv.imshow("G"+str(i),dst)
        temp = dst.copy()
    return G

#拉普拉斯金字塔
def demo1(src2):
    G = demo(src2)
    level = len(G)
    for i in range(level-1,-1,-1):
        if(i-1)<0:
            expand = cv.pyrUp(G[i],dstsize=src2.shape[:2])
            L = cv.subtract(src2,expand)
            cv.imshow("l" + str(i), L)
        else:
            expand = cv.pyrUp(G[i],dstsize=G[i-1].shape[:2])
            L=cv.subtract(G[i-1],expand)
            cv.imshow("l"+str(i),L)


demo(src)
demo1(src)
cv.waitKey(0)
cv.destroyAllWindows()
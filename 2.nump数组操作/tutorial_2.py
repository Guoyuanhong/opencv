import cv2 as cv
import numpy as np

def access_pixles(image):   #遍历每个像素修改他的值
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]

    for row in range(height):
        for col in range (width):
            for c in range(channel):
                pv = image[row,col,c]
                image[row, col, c] = 255 - pv   #像素取反
    cv.imshow("new",image)

def convers(image): #比上面遍历修改更快
    img = cv.bitwise_not(image)
    cv.imshow("convers",img)

def creat_image():#自定义多通道图片
    img=np.zeros([400,400,3],np.uint8)  #zeros:double类零矩阵  创建400*400 3个通道的矩阵图像 参数时classname为uint8
    img[:,:,0] = np.ones([400,400])*255 #ones([400, 400])是创建一个400*400的全1矩阵，*255即是全255矩阵 并将这个矩阵的值赋给img的第一维
    #img[:,:,1] = np.ones([400,400])*255
    #img[:,:,2] = np.ones([400,400])*255
    cv.imshow("creat",img)

def creat_simple():#定义单通道图片
    img=np.ones([400,400,1],np.uint8)
    img=img*255         #0-255从黑到白
    cv.imshow("simple",img)

rsc = cv.imread("/home/gyh/opencv/lena.jpg")

t1=cv.getTickCount()    #记录时间，毫秒级别的计时函数,记录了系统启动以来的时间毫秒

#cv.imshow("old",rsc)
#access_pixles(rsc)
convers(rsc)
#creat_image()
creat_simple()

t2=cv.getTickCount()    #运行完函数再次记录时间
time = (t2-t1)/cv.getTickFrequency()    #getTickFrequency用于返回CPU的频率,就是每秒的计时周期数
print("用时：%s ms" %(time*1000))

cv.waitKey(0)
cv.destroyAllWindows()
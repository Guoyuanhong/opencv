# -*- coding=GBK -*-
import cv2 as cv
import numpy as np

def image_info(image):
    print(image.shape)  #打印图形形状的值
    print(image.size)   #图片大小
    print(type(image))  #numpy类型的数组
    print(image.dtype)  #图像类型
    image_data = np.array(image)
    print(image_data)   #打印图像矩阵

def vedio_read():           #由于没摄像头就不调用这个函数了
    capture = cv.VideoCapture("/home/gyh/opencv/demo_video.mp4")  # 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
    while True:
        ret, frame = capture.read()  # 读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
        frame = cv.flip(frame,1)  # 翻转 0:上下颠倒 大于0水平颠倒   小于180旋转
        cv.imshow("video", frame)
        c = cv.waitKey(50)
        if c==27:                   #按esc退出
            break

src = cv.imread(r"1.jpg")   #读取一张图片
cv.namedWindow("output",cv.WINDOW_NORMAL)  #打开一个窗口，0或者cv.WINDOW_NORMAL代表可以改变窗口大小,
                                            # 不写或cv.WINDOW_AUTOSIZE则不可改变大小
cv.imshow("output",src)     #在刚才的窗口上显示
image_info(src)
vedio_read()
cv.imwrite("copy_1.jpg",src)    #写入图片
cv.waitKey(0)               #窗口显示时间0代表一直显示
cv.destroyAllWindows()      #删除建立的窗口
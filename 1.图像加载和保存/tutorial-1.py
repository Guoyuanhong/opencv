# -*- coding=GBK -*-
import cv2 as cv
import numpy as np

def image_info(image):
    print(image.shape)  #��ӡͼ����״��ֵ
    print(image.size)   #ͼƬ��С
    print(type(image))  #numpy���͵�����
    print(image.dtype)  #ͼ������
    image_data = np.array(image)
    print(image_data)   #��ӡͼ�����

def vedio_read():           #����û����ͷ�Ͳ��������������
    capture = cv.VideoCapture("/home/gyh/opencv/demo_video.mp4")  # ������ͷ��0��������豸id������ж������ͷ����������������ֵ
    while True:
        ret, frame = capture.read()  # ��ȡ����ͷ,���ܷ���������������һ��������bool�͵�ret����ֵΪTrue��False��������û�ж���ͼƬ���ڶ���������frame���ǵ�ǰ��ȡһ֡��ͼƬ
        frame = cv.flip(frame,1)  # ��ת 0:���µߵ� ����0ˮƽ�ߵ�   С��180��ת
        cv.imshow("video", frame)
        c = cv.waitKey(50)
        if c==27:                   #��esc�˳�
            break

src = cv.imread(r"1.jpg")   #��ȡһ��ͼƬ
cv.namedWindow("output",cv.WINDOW_NORMAL)  #��һ�����ڣ�0����cv.WINDOW_NORMAL������Ըı䴰�ڴ�С,
                                            # ��д��cv.WINDOW_AUTOSIZE�򲻿ɸı��С
cv.imshow("output",src)     #�ڸղŵĴ�������ʾ
image_info(src)
vedio_read()
cv.imwrite("copy_1.jpg",src)    #д��ͼƬ
cv.waitKey(0)               #������ʾʱ��0����һֱ��ʾ
cv.destroyAllWindows()      #ɾ�������Ĵ���
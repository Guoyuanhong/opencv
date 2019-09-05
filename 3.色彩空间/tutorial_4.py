import cv2 as cv
import numpy as np

#追踪图形中的某个颜色

def demo():
    capture = cv.VideoCapture("/home/gyh/workplace/github/opencv/demo_video.mp4")    #打开视频

    while True:
        ret , frame = capture.read()    #读取视频
        if not ret:
            break
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)   #转换为shv
        lower = np.array([0,0,221])     #设置追踪颜色的低值
        upper = np.array([180,30,225])  #设置追踪颜色的高值
        mask = cv.inRange(hsv,lower,upper)  #调节图像颜色信息（H）、饱和度（S）、亮度（V）区间，选择白色区域
        cv.imshow("mask",mask)
        if cv.waitKey(40)& 0xff == ord('q'):
            break

demo()
cv.waitKey(0)
cv.destroyAllWindows()
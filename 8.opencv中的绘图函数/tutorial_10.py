import cv2 as cv
import numpy as np

img = np.zeros((400,400,3),np.uint8)
#画直线
#cv.line(img,(0,0),(100,100),(255,0,0))
#画圆
#cv.circle(img,(50,50),10,(255,0,0),-1)
#画矩形
cv.rectangle(img,(50,50),(100,100),(0,255,0),1) #1,表示线条粗细，如果-1，表示填充闭合曲线，园和椭圆也是


cv.imshow("img",img)
cv.waitKey(0)
cv.destroyAllWindows()
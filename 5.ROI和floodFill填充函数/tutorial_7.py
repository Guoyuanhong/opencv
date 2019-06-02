import cv2 as cv
import numpy as np

def ROI(src):
    img = src.copy()
    h,w = img.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8)
    cv.floodFill(img,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("img",img)
    """
    floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)
    Iimage:输入图像，可以是一通道或者是三通道。
    mask:该版本特有的掩膜。 单通道，8位，在长宽上都比原图像image多2个像素点。另外，当flag为FLOORFILL_MAK_ONLY时，只会填充mask中数值为0的区域。
    seedPoint:漫水填充的种子点，即起始点。
    newVal:被填充的像素点新的像素值
    loDiff：表示当前的观察点像素值与其相邻区域像素值或待加入该区域的像素之间的亮度或颜色之间负差的最大值。
    upDiff:表示当前的观察点像素值与其相邻区域像素值或待加入该区域的像素之间的亮度或颜色之间负差的最小值。
    flag：
    当为CV_FLOODFILL_FIXED_RANGE 时，待处理的像素点与种子点作比较，在范围之内，则填充此像素 。（改变图像）
    CV_FLOODFILL_MASK_ONLY 此位设置填充的对像， 若设置此位，则mask不能为空，此时，函数不填充原始图像img，而是填充掩码图像.
    """

def mask_only():
    img = np.ones([200,200,3],np.uint8)
    img[50:100,50:100,:]=255
    cv.imshow("img",img)
    mask= np.ones([202,202,1],np.uint8)
    mask[50:101,50:101]=0
    cv.floodFill(img,mask,(75,75),(0,0,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow("mask_only",img)

src = cv.imread("2.jpg")

#ROI(src)
mask_only()

cv.waitKey(0)
cv.destroyAllWindows()
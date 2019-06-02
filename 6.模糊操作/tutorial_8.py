import cv2 as cv
import numpy as np

def mohu(image):
    #均值模糊   卷积框覆盖区域所有像素的平均值来代替中心元素
    img1=cv.blur(image,(5,5))
    #src：要处理的原图像
    #ksize: 周围关联的像素的范围：代码中（5，5）就是9*5的大小，就是计算这些范围内的均值来确定中心位置的大小
    cv.imshow("blur",img1)

    #中值模糊   去除椒盐噪声,用与卷积框对应像素的中值来替代中心像素的值
    img2=cv.medianBlur(image,5)
    #ksize与blur()函数不同，不是矩阵，而是一个数字，例如为5，就表示了5*5的方阵
    cv.imshow("medianblur",img2)

    #高斯模糊   高斯滤波器是求中心点邻近区域像素的高斯加权平均值
    img3=cv.GaussianBlur(image,(5,5),0)
    #是指根据窗口大小( 5,5 )来计算高斯函数标准差，越大越模糊,sigmaX：标准差
    cv.imshow("gauss",img3)

    #双边高斯边缘滤波（EPF）
    img4 = cv.bilateralFilter(image,0,100,50)
    cv.imshow("img4",img4)
    """
    4.双边滤波函数bilateralFilter():定义：bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
    d：邻域直径 可由后面两个值计算
    sigmaColor：颜色标准差
    sigmaSpace：空间标准差
    
    对于高斯滤波，仅用空间距离的权值系数核与图像卷积后，确定中心点的灰度值。
    即认为离中心点越近的点，其权重系数越大。
    
    双边滤波中加入了对灰度信息的权重，即在邻域内，
    灰度值越接近中心点灰度值的点的权重更大，灰度值相差大的点权重越小。
    此权重大小，则由值域高斯函数确定。
    
    Sigma越大，边缘越模糊，极限情况为simga无穷大，值域系数近似相等（忽略常数时，将近为exp（0）= 1），与高斯模板（空间域模板）相乘后可认为等效于高斯滤波。
    Sigma越小，边缘越清晰，极限情况为simga无限接近0，值域系数近似相等（接近exp（-∞） =  0），与高斯模板（空间域模板）相乘后，可近似为系数皆相等，等效于源图像。
    """

def auto(image):
    kernel=np.ones((5,5),np.float)/25
    src1 = cv.filter2D(image,-1,kernel)     #自定义卷积核，此处kernel是均值核
    cv.imshow("src1",src1)
    kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)   #锐化的kernel
    src2 = cv.filter2D(src1, -1, kernel2)
    cv.imshow("src2",src2)
    """
    filter2D()：定义为filter2D(src,ddepth,kernel)
    ddepth：深度，输入值为-1时，目标图像和原图像深度保持一致
    kernel: 卷积核（或者是相关核）,一个单通道浮点型矩阵
    修改kernel矩阵即可实现不同的模糊
    """

src = cv.imread("/home/gyh/opencv/lena.jpg")
cv.imshow("old",src)
mohu(src)
#auto(src)
cv.waitKey(0)
cv.destroyAllWindows()
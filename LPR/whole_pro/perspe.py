import cv2
import numpy as np
from scipy.spatial import distance as dist


############################
#本文件作用:典型竖直透视角变化子库、典型水平透视角变化子库、竖直错切角变化子库、水平旋转角变化子库
############################

def order_points(pts):
    '''
    功能：将检测的轮廓四个点坐标按[左上, 右上, 右下, 左下]排序
    :param pts:轮廓对象的四个点坐标
    :return:[左上, 右上, 右下, 左下]坐标
    '''
    #根据输入的四个顶点坐标值，确定它们在矩形轮廓中位置
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    if (br[1] < tr[1]):
        (br, tr)=(tr, br)
    else:
        (br, tr) = (br, tr)
    return np.array([tl, tr, br, bl], dtype="float32")

def perspe(img):
    '''
    功能：变形图像修复
    :param img: 原始图像
    :return: 修复后的图像
    '''
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # binary 就是 thresh，cnts存放着检测到的所有轮廓的特征点，一个轮廓对应若干个特征点，
    binary,cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    docCnt=None
    if len(cnts) > 0:
        # 将轮廓大小按面积排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 选择最外边缘的作为轮廓
        c = cnts[1]
        epsilon = 0.02 * cv2.arcLength(c, True)
        # 对排序后的轮廓循环处理
        approx = cv2.approxPolyDP(c, epsilon, True)
        # print (approx)
        docCnt = approx
        if len(docCnt) == 4:
            for i in docCnt:
                # circle函数为在图像上作图，新建了一个图像用来演示四角选取
                cv2.circle(img1, (i[0][0],i[0][1]), 10, (255, 0, 0), -1)
                newimage = cv2.resize(img1, (800, 600), interpolation=cv2.INTER_CUBIC)
            # img1=cv2.resize(img1,(800,600))
            # cv2.imshow('qtest',newimage)
            # cv2.waitKey(0)
            rect = order_points(docCnt.reshape(4, 2))
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            if (maxWidth > 136):
                maxWidth = max(int(widthA), int(widthB))
            else:
                maxWidth = 125
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            if (maxHeight > 25):
                maxHeight = max(int(heightA), int(heightB))
            else:
                maxHeight = 25
            # 得到目标映射图像
            dst = np.array([[0, 0], [136, 0], [136, 36],
                        [0, 36]], dtype="float32")
            # 从原图像四点到目标映射图像的变换矩阵
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (136, 36))
        else:
            warped = img1[:136,:36]
        return warped

# if __name__ == "__main__":
#     I=cv2.imread("./img_test/60_0001.jpg")
#     I1=perspe(I)
#     cv2.imshow('test',I1)
#     cv2.waitKey(0)



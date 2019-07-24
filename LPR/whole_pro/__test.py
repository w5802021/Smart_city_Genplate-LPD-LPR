
"""
Author: youngorsu
Email : zhiyongsu@qq.com
Last edited: 2018.1.29
"""
# coding=utf-8
# coding=utf-8 #
# 以utf-8编码储存中文字符
############################################################################
#   本文件主要是实现Qt界面，类“HyperLprWindow”实现主窗口，并调用其他类和函数
#   包含函数：def SimpleRecognizePlateWithGui(image)：
#     包含类：class LicenseRecognizationThread(QThread):
#             class HyperLprImageView(QGraphicsView):
#             class HyperLprWindow(QMainWindow):
############################################################################

import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QTableWidget,
    QWidget,
    QAbstractItemView,
    QHeaderView,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSplitter,
    QFileDialog,
    QTableWidgetItem,
    QGraphicsRectItem,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QGraphicsSimpleTextItem,
    qApp,
    QAction,
    QApplication)
from PyQt5.QtGui import QIcon, QColor, QPainter, QImage, QPixmap, QPen, QBrush, QFont, QPalette, QKeySequence
from PyQt5.QtCore import Qt, QDir, QSize, QEventLoop, QThread, pyqtSignal
import pipline as pp
import perspe as psp
import cv2
import numpy as np
import time

import math

import shutil
import pandas as pd


draw_plate_in_image_enable = 1

plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

Son_k = ["典型竖直透视角变化子库", "典型水平透视角变化子库", "分辨率变化子库", "亮度不均匀变化子库", "平均亮度变化子库",
         "散焦模糊变化子库", "竖直错切角变化子库", "水平旋转角变化子库", "运动模糊变化子库"]

def SimpleRecognizePlateWithGui(image, path_save = "", name = ""):

        t0 = time.time()

        images = pp.detect.detectPlateRough(
            image, image.shape[0], top_bottom_padding_rate=0.1)

        res_set = []
        y_offset = 32
        for j, plate in enumerate(images):
            plate, rect, origin_plate = plate
            plate = cv2.resize(plate, (136, 36))

            cv2.imencode('.jpg', plate)[1].tofile("G:/RePicture/plate/" + str(j) + ".jpg")
            cv2.imencode('.jpg', origin_plate)[1].tofile("G:/RePicture/originplate/" + str(j) + ".jpg")

            t1 = time.time()

            # plate_type = pp.td.SimplePredict(plate)
            # plate_color = plateTypeName[plate_type]
            #
            # if (plate_type > 0) and (plate_type < 5):
            #     plate = cv2.bitwise_not(plate)

            # cv2.imencode('.jpg', plate)[1].tofile("G:/RePicture/bitwise_not/" + str(j) + ".jpg")
            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + plate.shape[0], 0:plate.shape[1]] = plate
            #     y_offset = y_offset + plate.shape[0] + 4

            image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)
            cv2.imencode('.jpg', image_rgb)[1].tofile("G:/RePicture/精定位后/" + str(j) + ".jpg")
            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + image_rgb.shape[0],
            #           0:image_rgb.shape[1]] = image_rgb
            #     y_offset = y_offset + image_rgb.shape[0] + 4

            image_rgb = pp.fv.finemappingVertical(image_rgb)

            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + image_rgb.shape[0],
            #           0:image_rgb.shape[1]] = image_rgb
            #     y_offset = y_offset + image_rgb.shape[0] + 4

            ################### 输出整张车牌 ####################
            if len(path_save) > 0:
                cv2.imencode('.jpg', image_rgb)[1].tofile(path_save + "/完整车牌/" + name)

            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + image_rgb.shape[0],
            #           0:image_rgb.shape[1]] = image_rgb
            #     y_offset = y_offset + image_rgb.shape[0] + 4

            # e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
            # print("e2e:", e2e_plate, e2e_confidence)

            plate_type = pp.td.SimplePredict(plate)
            plate_color = plateTypeName[plate_type]
            print("颜色：", plate_color)

            if (plate_type > 0) and (plate_type < 5):
                image_rgb = cv2.bitwise_not(image_rgb)

            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # print("校正", time.time() - t1, "s")

            t2 = time.time()
            val = pp.segmentation.slidingWindowsEval(image_gray)

            ################### 输出分割后的车牌 ####################
            if len(path_save) > 0 and len(val) > 0:
                for i in range(7):
                    # cv2.imencode('.jpg', val[0][i])[1].tofile(path_save + "/分割车牌/" + str(i) + "-" + name)
                    cv2.imencode('.jpg', val[0][i])[1].tofile("G:/RePicture/分割车牌/" + str(i) + "-" + name)
            # print val
            # print("分割和识别", time.time() - t2, "s")

            # res = ""
            # confidence = 0
            # if len(val) == 3:
            #     blocks, res, confidence = val
            #     if confidence / 7 > 0.7:
            #
            #         if draw_plate_in_image_enable == 1:
            #             image = pp.drawRectBox(image, rect, res)
            #             for i, block in enumerate(blocks):
            #                 block_ = cv2.resize(block, (24, 24))
            #                 block_ = cv2.cvtColor(block_, cv2.COLOR_GRAY2BGR)
            #                 image[j * 24:(j * 24) + 24, i *
            #                       24:(i * 24) + 24] = block_
            #                 if image[j * 24:(j * 24) + 24,
            #                          i * 24:(i * 24) + 24].shape == block_.shape:
            #                     pass
            #
            # res_set.append([res,
            #                 confidence / 7,
            #                 rect,
            #                 plate_color,
            #                 e2e_plate,
            #                 e2e_confidence,
            #                 len(blocks)])
            # print("seg:", res, confidence/7)
        # print(time.time() - t0, "s")
        print("---------------------------------")
        return image, res_set

# ***************************************************************************************
# 生成卷积核和锚点
def genaratePsf(length, angle):
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1
    # 模糊核大小
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1是左上角的权值较大，越往右下角权值越小的核。
    # 这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i * i + j * j)
            if rad >= (length/2) and math.fabs(psf1[i][j]) <= psfwdt:
                temp = (length/2) - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
                psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    # 运动方向是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    # 运动方向是往右上角移动，锚点一个在右上角
    # 同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle < 90 and angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0:  # 同理：往右下角移动
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)

    elif anchor < -90:  # 同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
        psf1 = psf1 / psf1.sum()
    return psf1, anchor

# 使用范例
# kernel, anchor = genaratePsf(20, 40)
# motion_blur = cv2.filter2D(image, -1, kernel, anchor=anchor)
# ***************************************************************************************

# 产生运动模糊图像
def motion_blur(image, degree=10, angle=20):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

# 使用范例
# motion_out = motion_blur(image, 20, 40)

# 产生散焦模糊图像
# cv2.GaussianBlur(image, ksize=(degree, degree), sigmaX=0, sigmaY=0)
# ***************************************************************************************

import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2
from skimage.measure import compare_ssim

# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if abs(slope_tan) <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def inverse(input, PSF, eps):  # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result


def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


def rad(x):
    return x * np.pi / 180

import predictionM as predm
if __name__ == '__main__':
        # path = "./img_test/功能评测图像库/车牌种类变化子库/教练车牌/"                # 粤C1557学
        # path = "./img_test/功能评测图像库/车牌种类变化子库/澳门出入境车牌/"                # 粤Z3810澳
        # path = "./img_test/功能评测图像库/车牌种类变化子库/新能源-小车牌/"                # 粤CD08828
        # path = "./img_test/功能评测图像库/车牌种类变化子库/新能源-大车牌/"                # 粤C00209D
        # path = "./Dataset/车牌种类变化子库/大型汽车前牌/"                  # 川B23523
        # path = "./Dataset/车牌种类变化子库/大型汽车后牌/"                 # 粤W07717       不出错，但没有操作：粤W07655
        # path = "./img_test/功能评测图像库/车牌种类变化子库/军用车牌/"     # GB34114
        # path = "./img_test/性能评测图像库/典型竖直透视角变化子库/竖直透视角60/"    # 60_0018
        # path = "./img_test/性能评测图像库/运动模糊变化子库/motion1/"    # 001_0001
        # path = "./img_test/功能评测图像库/省市简称变化子库/“新”牌/"    # 60_0018
        # name = "a.jpg"
        # path_to = "G:/RePicture"
        # image = cv2.imdecode(np.fromfile(path + name, dtype=np.uint8), -1)
        # image, res_set = SimpleRecognizePlateWithGui(image, path_to, name)

        # image = cv2.imdecode(np.fromfile("G:/RePicture/a.jpg", dtype=np.uint8), -1)
        # image_blurred = cv2.imdecode(np.fromfile("G:/RePicture/014_0001.jpg", dtype=np.uint8), -1)
        #
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)
        #
        # img_h = image.shape[0]
        # img_w = image.shape[1]
        # # graph.figure(1)
        # # graph.xlabel("Original Image")
        # # graph.gray()
        # # graph.imshow(image)  # 显示原图像
        #
        # graph.figure(2)
        # graph.gray()
        # # 进行运动模糊处理
        # PSF = motion_process((img_h, img_w), 60)
        # blurred = np.abs(make_blurred(image, PSF, 1e-3))
        #
        # score, diff = compare_ssim(image_blurred, blurred.astype("uint8"), full=True)
        #
        # graph.subplot(231)
        # graph.xlabel("Motion blurred")
        # graph.imshow(blurred)
        #
        #
        # result = inverse(image_blurred, PSF_max, 1e-3)  # 逆滤波
        # graph.subplot(232)
        # graph.xlabel("inverse deblurred")
        # graph.imshow(result)
        #
        # result = wiener(image_blurred, PSF_max, 1e-3)  # 维纳滤波
        # graph.subplot(233)
        # graph.xlabel("wiener deblurred(k=0.01)")
        # graph.imshow(result)
        #
        #
        # blurred_noisy = blurred + 0.1 * blurred.std() * \
        #                 np.random.standard_normal(blurred.shape)  # 添加噪声,standard_normal产生随机的函数
        #
        # graph.subplot(234)
        # graph.xlabel("motion & noisy blurred")
        # graph.imshow(blurred_noisy)  # 显示添加噪声且运动模糊的图像
        #
        # result = inverse(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行逆滤波
        # graph.subplot(235)
        # graph.xlabel("inverse deblurred")
        # graph.imshow(result)
        #
        # result = wiener(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行维纳滤波
        # graph.subplot(236)
        # graph.xlabel("wiener deblurred(k=0.01)")
        # graph.imshow(result)
        #
        # graph.show()

        a = predm.Pre7()

        a1, a2= a.Run("./OutDataset/完整车牌/典型水平透视角变化子库/水平透视角2/")
        print("a1:", a1)
        print("a2:", a2)

        a1, a2= a.Run("./OutDataset/完整车牌/典型水平透视角变化子库/水平透视角3/")
        print("a1:", a1)
        print("a2:", a2)






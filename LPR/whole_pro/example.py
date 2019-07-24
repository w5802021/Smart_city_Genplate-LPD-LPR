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
import shutil
import pandas as pd
import colourDetection as hc

import prediction8 as pred8

# ************************************************************************ #
#     本例程意在表述最终的大程序是如何调用模型识别车牌种类变化子库的
# ************************************************************************ #

if __name__ == '__main__':

    Ability_k = ["车牌种类变化子库", "省市简称变化子库"]

    A_caplat = ["大型汽车后牌", "低速车牌", "各类摩托车牌", "挂车牌", "拖拉机牌",
                "澳门出入境车牌", "香港出入境车牌",
                "领使馆车牌", "武警车牌",
                "大型汽车前牌", "小型汽车牌", "教练车牌", "警用车牌",
                "新能源-大车牌", "新能源-小车牌", "军用车牌"]

    pred = pred8.Pre8()                      # 定义对象

    Reg_path = "./OutDataset/完整车牌/车牌种类变化子库"

    Reg_temp = Reg_path .split('/')

    Reg_path_file_list = os.listdir(Reg_path)            # 获取该文件夹下面的子文件夹列表

    for i in range(0, len(Reg_path_file_list)):

        Reg_path_sondir = Reg_path + "/" + Reg_path_file_list[i]   # 获取子文件夹路径

        Reg_path_sondir_list = os.listdir(Reg_path_sondir)

        # 判断是否为车牌种类变化子库
        if Reg_temp[len(Reg_temp) - 1] == Ability_k[0]:

            # "大型汽车后牌", "低速车牌", "各类摩托车牌", "挂车牌", "拖拉机牌","澳门出入境车牌"，"香港出入境车牌"，"领使馆车牌"，"大型汽车前牌", "小型汽车牌",
            # "教练车牌", "警用车牌"
            if ((Reg_path_file_list[i] == A_caplat[0]) or (Reg_path_file_list[i] == A_caplat[1]) or
                    (Reg_path_file_list[i] == A_caplat[2]) or (Reg_path_file_list[i] == A_caplat[3]) or
                    (Reg_path_file_list[i] == A_caplat[4]) or (Reg_path_file_list[i] == A_caplat[5]) or
                    (Reg_path_file_list[i] == A_caplat[6]) or (Reg_path_file_list[i] == A_caplat[7]) or
                    (Reg_path_file_list[i] == A_caplat[9]) or (Reg_path_file_list[i] == A_caplat[10]) or
                    (Reg_path_file_list[i] == A_caplat[11]) or (Reg_path_file_list[i] == A_caplat[12]) or
                    (Reg_path_file_list[i] == A_caplat[15])):

                print("开始识别" + Reg_path_file_list[i])
                # 省略识别代码
                print("识别结束" + Reg_path_file_list[i])

            # "武警车牌","新能源-大车牌", "新能源-小车牌"
            else:
                print("开始识别" + Reg_path_file_list[i])

                # 识别车牌
                e2e_img_list, e2e_orin_list, accurace = pred.Run(Reg_path_sondir + "/")

                print("识别结束" + Reg_path_file_list[i])

        # 处理其他子库
        else:

            print("开始识别" + Reg_path_file_list[i])
            # 省略识别代码
            print("识别结束" + Reg_path_file_list[i])
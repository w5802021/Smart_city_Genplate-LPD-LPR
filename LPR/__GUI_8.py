"""
Project: SmartCity
Author : MysteriousTeam
University : SCUT
Team member: GUO + WEI + DUAN + LIANG
"""
# coding=utf-8

###############################################################################
#   本文件主要是实现Qt界面，类“HyperLprWindow”实现主窗口，并调用其他类和函数
#   包含函数：def SimpleRecognizePlateWithGui(image)：
#     包含类：class LicenseRecognizationThread(QThread):
#             class HyperLprImageView(QGraphicsView):
#             class HyperLprWindow(QMainWindow):
###############################################################################

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

import prediction7 as pred7
import prediction8 as pred8
import predictionX as predx
import predictionD as predd
import predictionH as predh
import predictionM as predm

draw_plate_in_image_enable = 1

Son_k = ["典型竖直透视角变化子库", "典型水平透视角变化子库", "分辨率变化子库", "亮度不均匀变化子库", "平均亮度变化子库",
         "散焦模糊变化子库", "竖直错切角变化子库", "水平旋转角变化子库", "运动模糊变化子库"]

Save_Segmentation_k = ["车牌种类变化子库", "省市简称变化子库"]

plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

def SimpleRecognizePlateWithGui(image, path_save = "", name = ""):

    print("name:", name)
    # 判断是是否为Son_k中的子库
    mask = False
    if len(path_save) > 0:
        temp_path = path_save.split('/')
        for i in range(len(Son_k)):
            if temp_path[len(temp_path) - 2] == Son_k[i]:
                mask = True
                break
    else:
        mask = False

    # 根据前面的判断，开始处理子库中的图片
    if mask == True:            # 处理性能子库图片
        res_set = []                         # 这是返回的一个参数，设置为空，不影响输出的图片
        image_m = psp.perspe(image)
        image = cv2.resize(image_m, (136, 36))
        ################### 输出整张车牌 ####################
        if len(path_save) > 0:
            pp.cache.verticalMappingToFolder(image, path_save, name)

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        val = pp.segmentation.slidingWindowsEval(image_gray)

        ################### 输出分割后的车牌（包括训练集和测试集） ####################
        if len(path_save) > 0 and len(val) > 0:
            pp.cache.verticalMappingToFolder_segmentation(val[0], path_save, name)  # 保存测试集
            pp.cache.Perform_save_segmentation_train(val[0], path_save, name)       # 保存训练集

    else:                       # 处理功能子库图片
        t0 = time.time()

        images = pp.detect.detectPlateRough(
            image, image.shape[0], top_bottom_padding_rate=0.1)

        res_set = []
        y_offset = 32

        for j, plate in enumerate(images):
            plate, rect, origin_plate = plate

            # plate = cv2.resize(plate, (136, 36))
            t1 = time.time()

            # 通过输出数字，然后在plateTypeName中匹配识别颜色 #
            plate_type = pp.td.SimplePredict(plate)
            # plate_color = plateTypeName[plate_type]                          # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

            if (plate_type > 0) and (plate_type < 5):
                plate = cv2.bitwise_not(plate)                               # 图片效果反转，白变黑、黑变白

            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + plate.shape[0], 0:plate.shape[1]] = plate
            #     y_offset = y_offset + plate.shape[0] + 4

            # 实现精定位 #
            # image_rgb = pp.fm.findContoursAndDrawBoundingBox(plate)

            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + image_rgb.shape[0],
            #           0:image_rgb.shape[1]] = image_rgb
            #     y_offset = y_offset + image_rgb.shape[0] + 4

            image_rgb = pp.fv.finemappingVertical(plate)   # image_rgb

            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + image_rgb.shape[0],
            #           0:image_rgb.shape[1]] = image_rgb
            #     y_offset = y_offset + image_rgb.shape[0] + 4

            ################### 输出整张车牌 ####################
            if len(path_save) > 0:
                pp.cache.verticalMappingToFolder(image_rgb, path_save, name, plate_type)

            # if draw_plate_in_image_enable == 1:
            #     image[y_offset:y_offset + image_rgb.shape[0],
            #           0:image_rgb.shape[1]] = image_rgb
            #     y_offset = y_offset + image_rgb.shape[0] + 4
            #
            # e2e_plate, e2e_confidence = pp.e2e.recognizeOne(image_rgb)
            # print("e2e:", e2e_plate, e2e_confidence)

            # image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            #
            # print("校正", time.time() - t1, "s")
            #
            # t2 = time.time()
            # val = pp.segmentation.slidingWindowsEval(image_gray)
            #
            # ################### 输出分割后的车牌（包括训练集和测试集）####################
            # if len(path_save) > 0 and len(val) > 0:
            #     pp.cache.verticalMappingToFolder_segmentation(val[0], path_save, name)    # 保存测试集
            #
            # # ### 判断是否为性能子库，然后保存分割训练数据集 ###
            # ability_perform_flag = False
            #
            # if len(path_save) > 0 and len(val) > 0:
            #     for i in range(len(Save_Segmentation_k)):
            #         if temp_path[len(temp_path) - 2] == Save_Segmentation_k[i]:
            #             ability_perform_flag = True
            #             break
            #     if ability_perform_flag == True:
            #         pp.cache.Ability_save_segmentation_train(val[0], path_save, name)    # 保存训练集
            #     else:
            #         pp.cache.Perform_save_segmentation_train(val[0], path_save, name)    # 保存训练集
            #
            # # print val
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


class LicenseRecognizationThread(QThread):                         # 继承基类QThread

    recognization_done_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hyperlpr_dir_path = ""
        self.filenames = []

    def set_parameter(self, filename_list, path):
        self.hyperlpr_dir_path = path
        self.filenames = filename_list

    def run(self):
       # while True:
       #     time.sleep(1)
            if len(self.hyperlpr_dir_path) > 0:
                for i in range(0, len(self.filenames)):
                    path = os.path.join(
                        self.hyperlpr_dir_path, self.filenames[i])
                    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                    image, res_set = SimpleRecognizePlateWithGui(image, self.hyperlpr_dir_path, self.filenames[i])
                    self.recognization_done_signal.emit([i, res_set])                              # recognization_done_signal绑定了recognization_done_slot

                self.hyperlpr_dir_path = ""


class HyperLprImageView(QGraphicsView):

    def __init__(self):

        super().__init__()

        self.init_ui()

    def init_ui(self):

        scene = QGraphicsScene()
        scene.setBackgroundBrush(QColor(100, 100, 100))
        scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)

        scene.setSceneRect(scene.itemsBoundingRect())

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)

        self.frame_item = QGraphicsPixmapItem()

        self.text_item_offset = 0
        self.rect_item_array = []
        self.text_item_array = []
        for i in range(0, 5):
            rect_item = QGraphicsRectItem()
            rect_item.setVisible(False)
            rect_item.setZValue(20.0)
            rect_item.setPen(QPen(Qt.red, 5))
            rect_item.setRect(20, 20, 20, 20)
            scene.addItem(rect_item)
            self.rect_item_array.append(rect_item)
            text_item = QGraphicsSimpleTextItem("")
            text_item.setBrush(QBrush(Qt.red))
            text_item.setZValue(20.0)
            text_item.setPos(10, 50)
            text_item.setFont(QFont("黑体", 24))
            text_item.setVisible(False)
            scene.addItem(text_item)
            self.text_item_array.append(text_item)

        scene.addItem(self.frame_item)

        self.curr_factor = 1.0

        self.setScene(scene)

    def resetRectText(self, res_set):
        max_no = len(res_set)

        if max_no > 5:
            max_no = 5

        for i in range(0, 5):
            if i < max_no:
                curr_rect = res_set[i][2]
                self.rect_item_array[i].setRect(int(curr_rect[0]), int(
                    curr_rect[1]), int(curr_rect[2]), int(curr_rect[3]))
                self.rect_item_array[i].setVisible(True)

                self.text_item_array[i].setText(
                    res_set[i][4] + " " + res_set[i][3])
                self.text_item_array[i].setPos(
                    int(curr_rect[0]), int(curr_rect[1]) - 48)
                self.text_item_array[i].setVisible(True)
            else:
                self.text_item_array[i].setVisible(False)
                self.rect_item_array[i].setVisible(False)

    def wheelEvent(self, event):
        factor = event.angleDelta().y() / 120.0
        if event.angleDelta().y() / 120.0 > 0:
            factor = 1.08
        else:
            factor = 0.92

        if self.curr_factor > 0.1 and self.curr_factor < 10:
            self.curr_factor = self.curr_factor * factor
            self.scale(factor, factor)

    def resetPixmap(self, image):

        self.frame_item.setPixmap(QPixmap.fromImage(image))


################################################
#                   主窗口类
################################################

class HyperLprWindow(QMainWindow):

    start_init_signal = pyqtSignal()                                  # 定义一个start_init_signal信号，该信号没有参数

    def __init__(self):                     # 构造函数，魔法方法

        super().__init__()                   # 调用超类构造函数
        self.initUI()

    def initUI(self):

        self.statusBar().showMessage('Ready')

        self.left_action = QAction('上一个', self)
        self.left_action.setShortcut(QKeySequence.MoveToPreviousChar)
        self.left_action.triggered.connect(self.analyze_last_one_image)                 # 绑定本类函数analyze_last_one_image

        self.right_action = QAction('下一个', self)
        self.right_action.setShortcut(QKeySequence.MoveToNextChar)
        self.right_action.triggered.connect(self.analyze_next_one_image)               # 绑定本类函数analyze_next_one_image

        self.rename_image_action = QAction('合并结果文件', self)
        self.rename_image_action.setShortcut(QKeySequence.MoveToPreviousLine)
        self.rename_image_action.triggered.connect(self.rename_current_image_with_info)      # 绑定本类函数rename_current_image_with_info

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Function')
        fileMenu.addAction(self.left_action)
        fileMenu.addAction(self.right_action)
        fileMenu.addAction(self.rename_image_action)

        self.image_window_view = HyperLprImageView()               # HyperLprImageView是一个类

        table_widget_header_labels = [
            "文件名",
            "分割识别",
            "置信度",
            "颜色",
            "E2E识别",
            "E2E置信度"]

        self.hyperlpr_tableview = QTableWidget(                              # 定义一个显示数据表格的控件
            0, len(table_widget_header_labels))
        self.hyperlpr_tableview.setHorizontalHeaderLabels(
            table_widget_header_labels)

        self.hyperlpr_tableview.setSelectionBehavior(
            QAbstractItemView.SelectItems)
        self.hyperlpr_tableview.setSelectionMode(
            QAbstractItemView.SingleSelection)
        self.hyperlpr_tableview.setEditTriggers(
            QAbstractItemView.NoEditTriggers)
        self.hyperlpr_tableview.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.hyperlpr_tableview.setEditTriggers(
            QAbstractItemView.NoEditTriggers)

        self.hyperlpr_tableview.cellClicked.connect(                        # 绑定本类函数recognize_one_license_plate
            self.recognize_one_license_plate)

        self.left_button = QPushButton("<")
        self.left_button.setFixedWidth(60)
        self.right_button = QPushButton(">")
        self.right_button.setFixedWidth(60)
        self.left_button.setEnabled(False)
        self.right_button.setEnabled(False)
        self.left_button.clicked.connect(self.analyze_last_one_image)          # 前面也有绑定代码
        self.right_button.clicked.connect(self.analyze_next_one_image)         # 前面也有绑定代码
        left_right_layout = QHBoxLayout()
        left_right_layout.addStretch()
        left_right_layout.addWidget(self.left_button)
        left_right_layout.addStretch()
        left_right_layout.addWidget(self.right_button)
        left_right_layout.addStretch()

        self.location_label = QLabel("车牌目录", self)
        self.location_text = QLineEdit(self)
        self.location_text.setEnabled(False)
        #self.location_text.setFixedWidth(300)
        self.location_button = QPushButton("...")
        self.location_button.clicked.connect(self.select_new_dir)               # 绑定本类函数select_new_dir，与选择数据集图片库有关

        self.location_layout = QHBoxLayout()
        self.location_layout.addWidget(self.location_label)
        self.location_layout.addWidget(self.location_text)
        self.location_layout.addWidget(self.location_button)
        self.location_layout.addStretch()

        self.check_box = QCheckBox("与文件名比较车牌")
        self.check_box.setChecked(True)

        self.update_file_path_button = QPushButton('批量识别')
        self.update_file_path_button.clicked.connect(
            self.batch_recognize_all_images)                                    # 绑定本类函数batch_recognize_all_images

        self.update_file_path_layout = QHBoxLayout()
        self.update_file_path_layout.addWidget(self.check_box)
        self.update_file_path_layout.addWidget(self.update_file_path_button)
        self.update_file_path_layout.addStretch()

        self.save_as_e2e_filename_button = QPushButton("合并结果文件")
        self.save_as_e2e_filename_button.setEnabled(False)
        self.save_as_e2e_filename_button.clicked.connect(self.rename_current_image_with_info)                # 前面已经有绑定代码
        self.save_layout = QHBoxLayout()
        self.save_layout.addWidget(self.save_as_e2e_filename_button)
        self.save_layout.addStretch()

        self.top_layout = QVBoxLayout()
        self.top_layout.addLayout(left_right_layout)
        self.top_layout.addLayout(self.location_layout)
        self.top_layout.addLayout(self.update_file_path_layout)
        self.top_layout.addLayout(self.save_layout)

        function_groupbox = QGroupBox("功能区")
        function_groupbox.setLayout(self.top_layout)

        license_plate_image_label = QLabel("车牌图")
        self.license_plate_widget = QLabel("")

        block_image_label = QLabel("分割图")
        self.block_plate_widget = QLabel("")

        filename_label = QLabel("文件名：")
        self.filename_edit = QLineEdit()

        segmentation_recognition_label = QLabel("分割识别：")
        self.segmentation_recognition_edit = QLineEdit()
        self.segmentation_recognition_edit.setFont(QFont("黑体", 24, QFont.Bold))
        # self.segmentation_recognition_edit.setStyleSheet("color:red")

        confidence_label = QLabel("分割识别\n置信度")
        self.confidence_edit = QLineEdit()
        #self.confidence_edit.setFont(QFont("黑体", 24, QFont.Bold))
        # self.confidence_edit.setStyleSheet("color:red")

        plate_color_label = QLabel("车牌颜色")
        self.plate_color_edit = QLineEdit()
        self.plate_color_edit.setFont(QFont("黑体", 24, QFont.Bold))
        # self.plate_color_edit.setStyleSheet("color:red")

        e2e_recognization_label = QLabel("e2e识别：")
        self.e2e_recognization_edit = QLineEdit()
        self.e2e_recognization_edit.setFont(QFont("黑体", 24, QFont.Bold))
        # self.e2e_recognization_edit.setStyleSheet("color:red")

        e2e_confidence_label = QLabel("e2e置信度")
        self.e2e_confidence_edit = QLineEdit()
        #self.e2e_confidence_edit.setFont(QFont("黑体", 24, QFont.Bold))
        # self.e2e_confidence_edit.setStyleSheet("color:red")

        info_gridlayout = QGridLayout()
        line_index = 0
        info_gridlayout.addWidget(filename_label, line_index, 0)
        info_gridlayout.addWidget(self.filename_edit, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(license_plate_image_label, line_index, 0)
        info_gridlayout.addWidget(self.license_plate_widget, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(e2e_recognization_label, line_index, 0)
        info_gridlayout.addWidget(self.e2e_recognization_edit, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(
            segmentation_recognition_label, line_index, 0)
        info_gridlayout.addWidget(
            self.segmentation_recognition_edit, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(plate_color_label, line_index, 0)
        info_gridlayout.addWidget(self.plate_color_edit, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(block_image_label, line_index, 0)
        info_gridlayout.addWidget(self.block_plate_widget, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(confidence_label, line_index, 0)
        info_gridlayout.addWidget(self.confidence_edit, line_index, 1)
        line_index += 1
        info_gridlayout.addWidget(e2e_confidence_label, line_index, 0)
        info_gridlayout.addWidget(self.e2e_confidence_edit, line_index, 1)

        info_widget = QGroupBox("分割识别&e2e")

        info_widget.setLayout(info_gridlayout)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.hyperlpr_tableview)
        right_splitter.addWidget(function_groupbox)
        right_splitter.addWidget(info_widget)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(2, 1)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.image_window_view)                                    # image_window_view是HyperLprImageView类的一个实例
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)

        self.image_filename_list = []
        self.hyperlpr_dir_path = ""
        #################################
        # 自定义变量
        self.subfolder_path = []
        #################################
        self.segmentation_recognition_correct_number = 0
        self.color_correct_number = 0
        self.e2e_recognization_correct_number = 0
        self.current_row = 0

        self.batch_recognization_thread = LicenseRecognizationThread()                      # 定义一个LicenseRecognizationThread类实例
        self.batch_recognization_thread.recognization_done_signal.connect(
            self.recognization_done_slot)                                                   # 绑定本类函数recognization_done_slot
        self.batch_recognization_thread.start()

        self.start_init_signal.connect(self.read_path_and_show_one_image)                   # 绑定start_init_signal信号和函数read_path_and_show_one_image

        self.setCentralWidget(main_splitter)

        self.setWindowTitle("~神~神~秘~秘~车牌识别软件v1.0")

        self.start_init_signal.emit()                                                       # 发射信号start_init_signal

    # 读取并显示一张图片，在调用本类时自动调用本函数 #
    def read_path_and_show_one_image(self):
        hyperlpr_dir_info_filepath = "./img_test"                     #  QDir.homePath() + "/hyperlpr_dir_file"  # 读取工程文件夹下面的一张图片，地址可以自己设置
        if os.path.exists(hyperlpr_dir_info_filepath):
            # with open(hyperlpr_dir_info_filepath, 'r') as f:
            #    self.hyperlpr_dir_path = f.read()                                    # 读取"hyperlpr_dir_info_filepath"
            self.hyperlpr_dir_path = hyperlpr_dir_info_filepath

        if len(self.hyperlpr_dir_path) > 0:
            # self.reset_info_gui()
            self.location_text.setText(self.hyperlpr_dir_path)  # 将地址传递给“车牌目录”显示控件,并显示图片
            self.scan_files_with_new_dir(self.hyperlpr_dir_path)  # 此处就是为了获得image_filename_list，这个变量是个属于整个类的，所以没有显性使用
            self.fill_table_with_new_info()

        if len(self.image_filename_list) > 0:
            self.recognize_and_show_one_image(self.image_filename_list[0], 0)               # 设置为image_filename_list[0]，只识别第一张图


    # 读取图片库按钮的响应函数
    def select_new_dir(self):
        self.hyperlpr_dir_path = QFileDialog.getExistingDirectory(
            self, "读取文件夹", QDir.currentPath())

        if len(self.hyperlpr_dir_path) > 0:
            # hyperlpr_dir_info_filepath = QDir.homePath() + "/hyperlpr_dir_file"
            # with open(hyperlpr_dir_info_filepath, 'w') as f:
            #     f.write(self.hyperlpr_dir_path)                                           # 写入数据
            self.reset_info_gui()

    # 关联“合并结果文件”控件的函数
    def rename_current_image_with_info(self):

        result_flag = True
        csv_list = os.listdir("./OutFile")

        for i in range(0, len(csv_list)):
            if csv_list[i] == "result.csv":
                result_flag = False
        try:
            if result_flag == True:
                dir = ["车牌种类变化子库", "省市简称变化子库", "典型竖直透视角变化子库", "典型水平透视角变化子库",
                        "分辨率变化子库", "亮度不均匀变化子库", "平均亮度变化子库", "散焦模糊变化子库",
                        "竖直错切角变化子库", "水平旋转角变化子库", "运动模糊变化子库"]

                for i in range(0, len(dir)):
                    result = pd.read_csv("./OutFile/" + dir[i] + ".csv", encoding="utf_8_sig", engine='python')
                    Final_result = result[["车牌号", "车牌颜色", "测试文件名"]]
                    Final_result.to_csv("./OutFile/result.csv", encoding="utf_8_sig", index=False, header=False, mode='a+')

                print("合并完毕")
            else:
                print("你没有删除上次生成的result.csv文件（没有覆盖上次的文件）！ ！ ！")
        except:
            print("你的文件存放不符合规则,可能是少了文件！ ！ ！")

    # 显示读取的图片库路径，并获得含文件名的字符串
    def reset_info_gui(self):
        self.location_text.setText(self.hyperlpr_dir_path)                               # 将地址传递给“车牌目录”显示控件,并显示图片

        name_file_list = os.listdir(self.hyperlpr_dir_path)
        self.subfolder_path.clear()
        for i in range(0, len(name_file_list)):
            sub_path = self.hyperlpr_dir_path + "/" + name_file_list[i]
            if os.path.isdir(sub_path) == True:                                         # 滤出图片
                self.subfolder_path.append(sub_path)

        if len(self.subfolder_path) == 0:
           print("进入只包含图片子库")
           self.scan_files_with_new_dir(self.hyperlpr_dir_path)                             # 此处就是为了获得image_filename_list，这个变量是个属于整个类的，所以没有显性使用
           self.fill_table_with_new_info()

    # 遍历读取到的路径下的所有文件名称，并判断是否为.jpg或者.png图片
    def scan_files_with_new_dir(self, path):
        name_list = os.listdir(path)                                                   # 列出文件夹下所有的目录与文件，输出例如：['OUT', '汽车车牌识别图像库', '测试集']；或者['粤Z7087澳.jpg', '粤Z7426澳.jpg', '粤Z7442澳.jpg']
        self.image_filename_list.clear()
        for i in range(0, len(name_list)):
            if name_list[i].endswith(                                                  # endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
                    ".jpg") or name_list[i].endswith(".png"):
                self.image_filename_list.append(name_list[i])                          # image_filename_list包含所有.jpg和.png图片的名称

    # 更新表格控件显示的信息
    def fill_table_with_new_info(self):
        self.hyperlpr_tableview.clearContents()
        row_count = self.hyperlpr_tableview.rowCount()
        for i in range(row_count, -1, -1):
            self.hyperlpr_tableview.removeRow(i)

        for i in range(0, len(self.image_filename_list)):
            row = self.hyperlpr_tableview.rowCount()
            self.hyperlpr_tableview.insertRow(row)

            item0 = QTableWidgetItem()
            item0.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(row, 0, item0)
            self.hyperlpr_tableview.item(
                row, 0).setText(
                self.image_filename_list[i])

            item1 = QTableWidgetItem()
            item1.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(row, 1, item1)

            item2 = QTableWidgetItem()
            item2.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(row, 2, item2)

            item3 = QTableWidgetItem()
            item3.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(row, 3, item3)

            item4 = QTableWidgetItem()
            item4.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(row, 4, item4)

            item5 = QTableWidgetItem()
            item5.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(row, 5, item5)

        if len(self.image_filename_list) > 0:
            self.left_button.setEnabled(True)
            self.right_button.setEnabled(True)
            self.save_as_e2e_filename_button.setEnabled(True)

    def analyze_last_one_image(self):
        if self.current_row > 0:
            self.recognize_one_license_plate(self.current_row-1, 0)

    def analyze_next_one_image(self):
        if self.current_row < (len(self.image_filename_list)-1):
            self.recognize_one_license_plate(self.current_row + 1, 0)

    # 表格控件关联的函数，因为它，所以点击表格控件的每个格子都可以识别车牌
    # row, col会随着点击的表格位置而变化
    def recognize_one_license_plate(self, row, col):
        if col == 0 and row < len(self.image_filename_list):
            self.current_row = row
            self.recognize_and_show_one_image(
                self.image_filename_list[row], row)

    # 识别并显示一张车牌
    def recognize_and_show_one_image(self, image_filename_text, row):

        if image_filename_text.endswith(".jpg"):
            print(image_filename_text)                                                # 输出车牌图片名字
            path = os.path.join(self.hyperlpr_dir_path, image_filename_text)
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)              # fromfile()函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
                                                                                      # cv2.imdecode()函数从指定的内存缓存中读取数据，并把数据转换(解码)成图像格式;
            image, res_set = SimpleRecognizePlateWithGui(image)                       # SimpleRecognizePlateWithGui函数本文件最前面有定义
            img = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.shape[1] * image.shape[2],
                QImage.Format_RGB888)
            self.image_window_view.resetPixmap(img.rgbSwapped())                                   # image_window_view是HyperLprImageView类的一个实例
            self.image_window_view.resetRectText(res_set)

            if len(res_set) > 0:
                curr_rect = res_set[0][2]
                image_crop = image[int(curr_rect[1]):int(
                    curr_rect[1] + curr_rect[3]), int(curr_rect[0]):int(curr_rect[0] + curr_rect[2])]
                curr_plate = cv2.resize(image_crop, (204, 108))
                plate_img = QImage(
                    curr_plate.data,
                    curr_plate.shape[1],
                    curr_plate.shape[0],
                    curr_plate.shape[1] * curr_plate.shape[2],
                    QImage.Format_RGB888)
                self.license_plate_widget.setPixmap(
                    QPixmap.fromImage(plate_img.rgbSwapped()))

                # print(res_set[0][6])
                block_crop = image[0:24, 0:(24 * int(res_set[0][6]))]
                curr_block = cv2.resize(
                    block_crop, (24 * int(res_set[0][6]), 24))
                block_image = QImage(
                    curr_block.data,
                    curr_block.shape[1],
                    curr_block.shape[0],
                    curr_block.shape[1] * curr_block.shape[2],
                    QImage.Format_RGB888)
                self.block_plate_widget.setPixmap(
                    QPixmap.fromImage(block_image.rgbSwapped()))

                self.segmentation_recognition_edit.setText(res_set[0][0])
                if res_set[0][0] in image_filename_text:
                    self.segmentation_recognition_edit.setStyleSheet("color:black")
                else:
                    self.segmentation_recognition_edit.setStyleSheet("color:red")


                self.filename_edit.setText(image_filename_text)
                self.confidence_edit.setText("%.3f" % (float(res_set[0][1])))

                self.plate_color_edit.setText(res_set[0][3])
                if res_set[0][3] in image_filename_text:
                    self.plate_color_edit.setStyleSheet("color:black")
                else:
                    self.plate_color_edit.setStyleSheet("color:red")

                self.e2e_recognization_edit.setText(res_set[0][4])
                if res_set[0][4] in image_filename_text:
                    self.e2e_recognization_edit.setStyleSheet("color:black")
                else:
                    self.e2e_recognization_edit.setStyleSheet("color:red")

                self.e2e_confidence_edit.setText(
                    "%.3f" % (float(res_set[0][5])))
            else:
                self.license_plate_widget.clear()
                self.block_plate_widget.clear()
                self.segmentation_recognition_edit.setText("")
                self.filename_edit.setText(image_filename_text)
                self.confidence_edit.setText("")
                self.plate_color_edit.setText("")
                self.e2e_recognization_edit.setText("")
                self.e2e_confidence_edit.setText("")

            self.fill_table_widget_with_res_info(res_set, row)            # 更新信息

    def batch_recognize_all_images(self):
        # 先从img_test中提取车牌，再进行识别
        if len(self.subfolder_path) == 0:                         # 当直接进入只包含图片的路径时，进行处理
            print("进入只包含图片子库")
            self.segmentation_recognition_correct_number = 0
            self.color_correct_number = 0
            self.e2e_recognization_correct_number = 0
            self.batch_recognization_thread.set_parameter(                                # batch_recognization_thread是LicenseRecognizationThread类的一个实例
                self.image_filename_list, self.hyperlpr_dir_path)                         # 设置好批量处理的参数
            self.batch_recognization_thread.run()
        else:
            print("进入计分子库")

            orig_count = 0

            for i in range(0, len(self.subfolder_path)):          # 当进入计分子库路径时，进行处理
                self.scan_files_with_new_dir(self.subfolder_path[i])

                orig_count = orig_count + len(self.image_filename_list)

                self.fill_table_with_new_info()
                self.segmentation_recognition_correct_number = 0
                self.color_correct_number = 0
                self.e2e_recognization_correct_number = 0
                self.batch_recognization_thread.set_parameter(                                # batch_recognization_thread是LicenseRecognizationThread类的一个实例
                    self.image_filename_list, self.subfolder_path[i])                         # 设置好批量处理的参数
                self.batch_recognization_thread.run()

            print("车牌提取结束")

            # 进行识别
            print("开始识别")
            print("图片总数量：", orig_count)

            Ability_k = ["车牌种类变化子库", "省市简称变化子库"]

            A_caplat = ["大型汽车后牌", "低速车牌", "各类摩托车牌", "挂车牌", "拖拉机牌",
                        "澳门出入境车牌", "香港出入境车牌",
                        "领使馆车牌", "武警车牌",
                        "大型汽车前牌", "小型汽车牌", "教练车牌", "警用车牌",
                        "新能源-大车牌", "新能源-小车牌", "军用车牌"]

            Perfor_k = ["典型竖直透视角变化子库", "典型水平透视角变化子库", "分辨率变化子库", "亮度不均匀变化子库", "平均亮度变化子库",
                     "散焦模糊变化子库", "竖直错切角变化子库", "水平旋转角变化子库", "运动模糊变化子库"]

            Perfor_plate = ["粤A6503H","粤A177US","粤A3659D","粤A0RE33","粤A4102J","粤A5RW15","粤A9CE55","粤AC777X","粤A7AB78","粤A8716Z",
                            "粤AY658X","粤A8KS92","粤A0TH69","粤A5HR14","粤AL372R","粤A0GM68","粤A973FS","粤A6860U","粤A205ZE","粤A2KC31"]

            Reg_flag = False

            out_result_list = []                       # 存储每个计分子库的车牌识别结果
            out_color_list = []                        # 存储每个计分子库的车牌颜色识别结果
            out_filename_list = []                     # 存储每个计分子库的车牌名称
            Reg_count = 0

            if len(self.hyperlpr_dir_path) > 0:
                Reg_temp = self.hyperlpr_dir_path.split('/')

            for i in range(len(Ability_k)):                               # 判断处理的子库是否为功能子库
                if Reg_temp[len(Reg_temp) - 1] == Ability_k[i]:
                    Reg_flag = True
                    break

            Reg_path = "./OutDataset/完整车牌/" + Reg_temp[len(Reg_temp) - 1]  # 处理其他子库
            Reg_path_file_list = os.listdir(Reg_path)

            # 处理车牌种类变化子库
            if Reg_temp[len(Reg_temp) - 1] == Ability_k[0]:

                pred88 = pred8.Pre8()  # 定义一个调用7位模型识别车牌的对象

                for i in range(0, len(Reg_path_file_list)):
                    Reg_path_sondir = Reg_path + "/" + Reg_path_file_list[i]
                    Reg_path_sondir_list = os.listdir(Reg_path_sondir)

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
                        for j in range(0, len(Reg_path_sondir_list)):
                            img = cv2.imdecode(np.fromfile(Reg_path_sondir + "/" + Reg_path_sondir_list[j], dtype=np.uint8), -1)
                            # 对颜色进行识别
                            color_type = pp.td.SimplePredict(img)
                            img_color = plateTypeName[color_type]     # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

                            # 对车牌进行识别
                            if (color_type > 0) and (color_type < 5):
                                img = cv2.bitwise_not(img)

                            if (Reg_path_file_list[i] == A_caplat[11]) or (Reg_path_file_list[i] == A_caplat[12]):

                                e2e_img, e2e_confidence = pp.e2e.recognizeOne(img)

                            else:

                                e2e_img, e2e_confidence = pp.e2e.myself_recognizeOne(img)

                            if (e2e_img + ".jpg") == Reg_path_sondir_list[j]:
                                Reg_count = Reg_count + 1

                            filename_list_out = Reg_temp[len(Reg_temp) - 3] + "\\" + Reg_temp[len(Reg_temp) - 2] + "\\" + Reg_temp[len(Reg_temp) - 1] + "\\" + Reg_path_file_list[i] + "\\" + Reg_path_sondir_list[j]

                            out_color_list.append(img_color)
                            out_result_list.append(e2e_img)
                            out_filename_list.append(filename_list_out)

                    # "武警车牌","新能源-大车牌", "新能源-小车牌"
                    else:
                        print("开始识别" + Reg_path_file_list[i])

                        for j in range(0, len(Reg_path_sondir_list)):
                            img = cv2.imdecode(np.fromfile(Reg_path_sondir + "/" + Reg_path_sondir_list[j], dtype=np.uint8), -1)
                            # 对颜色进行识别
                            color_type = pp.td.SimplePredict(img)
                            img_color = plateTypeName[color_type]     # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

                            out_color_list.append(img_color)

                        # 识别车牌
                        e2e_img_list, e2e_orin_list, accurace = pred88.Run(Reg_path_sondir + "/")

                        Reg_count = Reg_count + round(len(Reg_path_sondir_list) * accurace)

                        out_result_list = out_result_list + e2e_img_list

                        for m in range(0, len(e2e_orin_list)):
                            filename_list_out = Reg_temp[len(Reg_temp) - 3] + "\\" + Reg_temp[len(Reg_temp) - 2] + "\\" + Reg_temp[len(Reg_temp) - 1] + "\\" + Reg_path_file_list[i] + "\\" + e2e_orin_list[m]
                            out_filename_list.append(filename_list_out)

            # 处理典型水平透视角变化子库
            elif Reg_temp[len(Reg_temp) - 1] == Perfor_k[1]:

                predh7 = predh.Pre7()  # 定义一个调用7位模型识别车牌的对象

                for i in range(0, len(Reg_path_file_list)):

                    print("开始识别" + Reg_path_file_list[i])
                    Reg_path_sondir = Reg_path + "/" + Reg_path_file_list[i]
                    Reg_path_sondir_list = os.listdir(Reg_path_sondir)

                    for j in range(0, len(Reg_path_sondir_list)):
                        img = cv2.imdecode(np.fromfile(Reg_path_sondir + "/" + Reg_path_sondir_list[j], dtype=np.uint8), -1)
                        # 对颜色进行识别
                        color_type = pp.td.SimplePredict(img)
                        img_color = plateTypeName[color_type]  # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

                        out_color_list.append(img_color)

                    # 识别车牌
                    e2e_img_list, e2e_orin_list = predh7.Run(Reg_path_sondir + "/")

                    for n in range(0, len(e2e_img_list)):
                        if e2e_img_list[n] == Perfor_plate[n]:
                            Reg_count = Reg_count + 1

                    out_result_list = out_result_list + e2e_img_list

                    for m in range(0, len(e2e_orin_list)):
                        filename_list_out = Reg_temp[len(Reg_temp) - 3] + "\\" + Reg_temp[len(Reg_temp) - 2] + "\\" + Reg_temp[
                            len(Reg_temp) - 1] + "\\" + Reg_path_file_list[i] + "\\" + e2e_orin_list[m]
                        out_filename_list.append(filename_list_out)

            # 处理散焦模糊子库
            elif Reg_temp[len(Reg_temp) - 1] == Perfor_k[5]:

                predd7 = predd.Pre7()  # 定义一个调用7位模型识别车牌的对象

                for i in range(0, len(Reg_path_file_list)):

                    print("开始识别" + Reg_path_file_list[i])
                    Reg_path_sondir = Reg_path + "/" + Reg_path_file_list[i]
                    Reg_path_sondir_list = os.listdir(Reg_path_sondir)

                    for j in range(0, len(Reg_path_sondir_list)):
                        img = cv2.imdecode(np.fromfile(Reg_path_sondir + "/" + Reg_path_sondir_list[j], dtype=np.uint8), -1)
                        # 对颜色进行识别
                        color_type = pp.td.SimplePredict(img)
                        img_color = plateTypeName[color_type]  # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

                        out_color_list.append(img_color)

                    # 识别车牌
                    e2e_img_list, e2e_orin_list = predd7.Run(Reg_path_sondir + "/")

                    for n in range(0, len(e2e_img_list)):
                        if e2e_img_list[n] == Perfor_plate[n]:
                            Reg_count = Reg_count + 1

                    out_result_list = out_result_list + e2e_img_list

                    for m in range(0, len(e2e_orin_list)):
                        filename_list_out = Reg_temp[len(Reg_temp) - 3] + "\\" + Reg_temp[len(Reg_temp) - 2] + "\\" + Reg_temp[
                            len(Reg_temp) - 1] + "\\" + Reg_path_file_list[i] + "\\" + e2e_orin_list[m]
                        out_filename_list.append(filename_list_out)

            # 处理运动模糊子库
            elif Reg_temp[len(Reg_temp) - 1] == Perfor_k[8]:
                predm7 = predm.Pre7()  # 定义一个调用7位模型识别车牌的对象

                for i in range(0, len(Reg_path_file_list)):

                    print("开始识别" + Reg_path_file_list[i])
                    Reg_path_sondir = Reg_path + "/" + Reg_path_file_list[i]
                    Reg_path_sondir_list = os.listdir(Reg_path_sondir)

                    for j in range(0, len(Reg_path_sondir_list)):
                        img = cv2.imdecode(np.fromfile(Reg_path_sondir + "/" + Reg_path_sondir_list[j], dtype=np.uint8), -1)
                        # 对颜色进行识别
                        color_type = pp.td.SimplePredict(img)
                        img_color = plateTypeName[color_type]  # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

                        out_color_list.append(img_color)

                    # 识别车牌
                    e2e_img_list, e2e_orin_list = predm7.Run(Reg_path_sondir + "/")

                    for n in range(0, len(e2e_img_list)):
                        if e2e_img_list[n] == Perfor_plate[n]:
                            Reg_count = Reg_count + 1

                    out_result_list = out_result_list + e2e_img_list

                    for m in range(0, len(e2e_orin_list)):
                        filename_list_out = Reg_temp[len(Reg_temp) - 3] + "\\" + Reg_temp[len(Reg_temp) - 2] + "\\" + Reg_temp[
                            len(Reg_temp) - 1] + "\\" + Reg_path_file_list[i] + "\\" + e2e_orin_list[m]
                        out_filename_list.append(filename_list_out)

            # 处理其他子库
            else:

                for i in range(0, len(Reg_path_file_list)):

                    print("开始识别" + Reg_path_file_list[i])
                    Reg_path_sondir = Reg_path + "/" + Reg_path_file_list[i]
                    Reg_path_sondir_list = os.listdir(Reg_path_sondir)

                    for j in range(0, len(Reg_path_sondir_list)):
                        img = cv2.imdecode(np.fromfile(Reg_path_sondir + "/" + Reg_path_sondir_list[j], dtype=np.uint8), -1)
                        color_type = pp.td.SimplePredict(img)
                        img_color = plateTypeName[color_type]  # plateTypeName = ["蓝", "黄", "绿", "白", "黑 "]

                        # 对车牌进行识别
                        if (color_type > 0) and (color_type < 5):
                            img = cv2.bitwise_not(img)

                        # e2e_img, e2e_confidence = pp.e2e.recognizeOne(img)
                        e2e_img, e2e_confidence = pp.e2e.myself_recognizeOne(img)

                        if Reg_flag == True:
                            if (e2e_img + ".jpg") == Reg_path_sondir_list[j]:
                                Reg_count = Reg_count + 1
                        else:
                            name_image = Reg_path_sondir_list[j]
                            flag_temp = int(name_image[len(name_image) - 6:len(name_image) - 4])
                            if e2e_img == Perfor_plate[flag_temp - 1]:
                                Reg_count = Reg_count + 1

                        filename_list_out = Reg_temp[len(Reg_temp) - 3] + "\\" + Reg_temp[len(Reg_temp) - 2] + "\\" + \
                                            Reg_temp[len(Reg_temp) - 1] + "\\" + Reg_path_file_list[i] + "\\" + \
                                            Reg_path_sondir_list[j]

                        out_color_list.append(img_color)
                        out_result_list.append(e2e_img)
                        out_filename_list.append(filename_list_out)

            Reg_rate = Reg_count / orig_count

            Reg_path_sondir_save = "./OutFile/" + Reg_temp[len(Reg_temp) - 1]

            rate_csv = np.array([[Reg_temp[len(Reg_temp) - 1]], [Reg_rate]]).T.tolist()
            rate_data = pd.DataFrame(columns=["车牌种类", "整牌识别率"], data=rate_csv)
            rate_data.to_csv(Reg_path_sondir_save + '-rate.csv', encoding="utf_8_sig")

            list_csv = np.array([out_result_list, out_color_list, out_filename_list]).T.tolist()
            name_csv = ['车牌号', '车牌颜色', '测试文件名']
            csv_data = pd.DataFrame(columns=name_csv, data=list_csv)
            csv_data.to_csv(Reg_path_sondir_save + '.csv', encoding="utf_8_sig")
            print("识别结束")

    # 作为LicenseRecognizationThread类带参信号recognization_done_signal的关联函数
    def recognization_done_slot(self, result_list):
        row = result_list[0]
        res_set = result_list[1]
        self.fill_table_widget_with_res_info(res_set, row)                           #fill_table_widget_with_res_info函数用于更新信息

        if row == len(self.image_filename_list) - 1:
            total_number = len(self.image_filename_list)

            row_count = self.hyperlpr_tableview.rowCount()
            if row_count > total_number:
                self.hyperlpr_tableview.removeRow(total_number)

            self.hyperlpr_tableview.insertRow(total_number)

            item0 = QTableWidgetItem()
            item0.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(total_number, 0, item0)
            self.hyperlpr_tableview.item(
                total_number, 0).setText(
                "统计结果")

            item1 = QTableWidgetItem()
            item1.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(total_number, 1, item1)
            self.hyperlpr_tableview.item(
                total_number,
                1).setText(
                "{0} / {1} = {2: .3f}".format(
                    self.segmentation_recognition_correct_number,
                    total_number,
                    self.segmentation_recognition_correct_number /
                    total_number))

            item2 = QTableWidgetItem()
            item2.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(total_number, 2, item2)

            item3 = QTableWidgetItem()
            item3.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(total_number, 3, item3)
            self.hyperlpr_tableview.item(
                total_number, 3).setText(
                "{0} / {1} = {2: .3f}".format(self.e2e_recognization_correct_number, total_number,
                                              self.e2e_recognization_correct_number / total_number))

            item4 = QTableWidgetItem()
            item4.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(total_number, 4, item4)
            self.hyperlpr_tableview.item(
                total_number, 4).setText(
                "{0} / {1} = {2: .3f}".format(self.color_correct_number, total_number,
                                              self.color_correct_number / total_number))

            item5 = QTableWidgetItem()
            item5.setTextAlignment(Qt.AlignCenter)
            self.hyperlpr_tableview.setItem(total_number, 5, item5)

    # 更新表格控件里面每张图片对应的相关信息
    def fill_table_widget_with_res_info(self, res_set, row):
        image_filename_text = self.image_filename_list[row]
        if len(res_set) > 0:

            self.hyperlpr_tableview.item(row, 1).setText(res_set[0][0])
            if res_set[0][0] in image_filename_text:
                self.hyperlpr_tableview.item(
                    row, 1).setForeground(
                    QBrush(
                        QColor(
                            0, 0, 255)))
                self.segmentation_recognition_correct_number += 1
            else:
                self.hyperlpr_tableview.item(
                    row, 1).setForeground(
                    QBrush(
                        QColor(
                            255, 0, 0)))

            self.hyperlpr_tableview.item(
                row, 2).setText(
                "%.3f" %
                (float(
                    res_set[0][1])))

            self.hyperlpr_tableview.item(row, 3).setText(res_set[0][3])
            if res_set[0][3] in image_filename_text:
                self.hyperlpr_tableview.item(
                    row, 3).setForeground(
                    QBrush(
                        QColor(
                            0, 0, 255)))
                self.color_correct_number += 1
            else:
                self.hyperlpr_tableview.item(
                    row, 3).setForeground(
                    QBrush(
                        QColor(
                            255, 0, 0)))

            self.hyperlpr_tableview.item(row, 4).setText(res_set[0][4])
            if res_set[0][4] in image_filename_text:
                self.hyperlpr_tableview.item(
                    row, 4).setForeground(
                    QBrush(
                        QColor(
                            0, 0, 255)))
                self.e2e_recognization_correct_number += 1
            else:
                self.hyperlpr_tableview.item(
                    row, 4).setForeground(
                    QBrush(
                        QColor(
                            255, 0, 0)))

            self.hyperlpr_tableview.item(
                row, 5).setText(
                "%.3f" %
                (float(
                    res_set[0][5])))


if __name__ == '__main__':

    app = QApplication(sys.argv)            # 获取命令行参数

    hyper_lpr_widow = HyperLprWindow()      # 创建一个类对象

    hyper_lpr_widow.showMaximized()         # HyperLprWindow继承了QMainWindow，QMainWindow继承了QWidget，QWidget包含函数showMaximized()

    sys.exit(app.exec_())

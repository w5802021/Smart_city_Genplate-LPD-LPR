#coding=utf-8
import numpy as np
import cv2
import time
import e2emodel as model

from keras.models import *
from keras.layers import *

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
         "粤", "桂","琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7",
         "8", "9", "A","B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
         "W", "X","Y", "Z", "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民",
         "航", "空"]

pred_model = model.construct_model("./model/ocr_plate_all_w_rnn_2.h5",)

def fastdecode(y_pred):
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, len(chars)+1)

    res = table_pred.argmax(axis=1)

    for i,one in enumerate(res):
        if one<len(chars) and (i==0 or (one!=res[i-1])):
            results+= chars[one]
            confidence+=table_pred[i][one]
    confidence/= len(results)
    return results, confidence

def recognizeOne(src):
    # x_tempx= cv2.imread(src)
    x_tempx = src
    # x_tempx = cv2.bitwise_not(x_tempx)
    x_temp = cv2.resize(x_tempx,(160,40))
    x_temp = x_temp.transpose(1, 0, 2)
    t0 = time.time()
    y_pred = pred_model.predict(np.array([x_temp]))
    y_pred = y_pred[:,2:,:]
    # plt.imshow(y_pred.reshape(16,66))
    # plt.show()

    #
    # cv2.imshow("x_temp",x_tempx)
    # cv2.waitKey(0)
    return fastdecode(y_pred)
#
#
# import os
#
# path = "/Users/yujinke/PycharmProjects/HyperLPR_Python_web/cache/finemapping"
# for filename in os.listdir(path):
#     if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
#         x = os.path.join(path,filename)
#         recognizeOne(x)
#         # print time.time() - t0
#
#         # cv2.imshow("x",x)
#         # cv2.waitKey()


#####################################################
#                  自己添加的代码
#####################################################
def model_seq_rec(model_path):
    '''
    模型搭建
    :param model_path:
    :return:
    '''
    width, height, n_len, n_class = 164, 48, 7, len(chars) + 1
    rnn_size = 256
    input_tensor = Input((width, height, 3))
    x = input_tensor
    base_conv = 32
    # 卷积池化层参数
    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    # 全连接层
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # GRU层
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    #
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)
    return base_model

modelSeqRec = model_seq_rec("./model/ocr_plate_all_gru.h5")

def myself_recognizeOne(src):
    x_tempx = src
    x_temp = cv2.resize(x_tempx,(164,48))
    x_temp = x_temp.transpose(1, 0, 2)
    y_pred = modelSeqRec.predict(np.array([x_temp]))
    y_pred = y_pred[:,2:,:]
    return fastdecode(y_pred)
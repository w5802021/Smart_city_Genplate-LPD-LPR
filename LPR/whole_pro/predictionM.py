#coding=utf-8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import mxnet as mx
import numpy as np
import cv2,random
import os

class Pre7():
    def __init__(self):
        # self.filepath = filepath
        self.network8 = './model/cnn-ocr-Motion'  # you can change the network here, such as cnn-ocr, cnn-ocr2, cnn-orc-test
        self.network7 = './model/cnn-ocr-Motion'
        self.chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂",
                 "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                 "A",
                 "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W",
                 "X",
                 "Y", "Z", "港", "澳", "领", "警"
                 ]

        self.chars_index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11,
                       "皖": 12,
                       "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22,
                       "贵": 23, "云": 24,
                       "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34,
                       "4": 35, "5": 36,
                       "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46,
                       "G": 47, "H": 48,
                       "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58,
                       "U": 59, "V": 60,
                       "W": 61, "X": 62, "Y": 63, "Z": 64, "港": 65, "澳": 66, "领": 67, "警": 68}
        self.length = 7
        _, self.arg_params, __ = mx.model.load_checkpoint(self.network7, 1)
        self.sym = self.getnet()

    def getnet(self):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('softmax_label')
        conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
        pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
        relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

        conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
        pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
        relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

        flatten = mx.symbol.Flatten(data=relu2)
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=120)
        fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc25 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc26 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc27 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25, fc26, fc27], dim=0)
        return mx.symbol.SoftmaxOutput(data=fc2, name="softmax")

    def getnet_8(self):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('softmax_label')
        conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
        pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
        relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

        conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
        pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
        relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

        flatten = mx.symbol.Flatten(data=relu2)
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=120)
        fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc25 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc26 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc27 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc28 = mx.symbol.FullyConnected(data=fc1, num_hidden=69)
        fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25, fc26, fc27, fc28], dim=0)
        return mx.symbol.SoftmaxOutput(data=fc2, name="softmax")

    def eachFile(self):
        filepath = self.filepath
        pathDir = os.listdir(filepath)
        Pname = []
        for allDir in pathDir:
            child = os.path.join('%s' % (allDir))
            Pname.append(child)
            # print (child)

        # selecting a sample to train the network
        samples_index = np.floor(np.random.random(1) * len(Pname))
        index = int(samples_index[0])
        # print ('index:',index)
        # print ('Pname[index]:',Pname[index])
        return Pname[index]

    def select_all(self):
        filepath = self.filepath
        pathDir = os.listdir(filepath)
        Pname = []
        for allDir in pathDir:
            child = os.path.join('%s' % (allDir))
            Pname.append(child)

        label_all = [None] * len(Pname)
        for j in range(len(Pname)):
            label = []
            name = Pname[j]
            for i in range(self.length):
                try:
                    label.append(self.chars_index[name[i]])
                except:
                    label.append(24)
            label_all[j] = label
        return Pname, label_all

    def get_file_above(self,file_list):
        rootdir = file_list
        Pname = []
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            Pname.append(path)
        return Pname

    def get_file_lower(self,D):
        Pname = []
        name_all = []
        for i in range(len(D)):
            x = self.get_file_above(D[i])
            Pname.append(x)
        Pname = np.array(Pname)

        # if os.path.isfile(path):
        # print (type(Pname),len(Pname),len(Pname[0]),Pname.shape)
        # print (Pname)
        for i in range(len(Pname)):
            for j in range(len(Pname[i])):
                # print (Pname[i][j])
                name_all.append(Pname[i][j])
        # print (name_all)
        return name_all

    def select_all_T(self,filepath):
        D = self.get_file_above(filepath)
        Pname = self.get_file_lower(D)
        Pname_T = []

        label_all = [None] * len(Pname)
        for j in range(len(Pname)):
            label = []
            name = Pname[j][-11:-4]
            Pname_T.append(name)
            for i in range(7):
                try:
                    label.append(self.chars_index[name[i]])
                except:
                    label.append(24)
            label_all[j] = label
        return Pname, label_all, Pname_T

    def TestRecognizeOne(self,img):
        # print('start')
        # img = cv2.resize(img,(120,30))
        # # cv2.imshow("img",img)
        #
        # # print (img.shape)
        # img = np.swapaxes(img,0,2)
        # img = np.swapaxes(img,1,2)
        # print (img[0].shape)
        batch_size = len(img)
        data_shape = [("data", (batch_size, 3, 30, 120))]
        input_shapes = dict(data_shape)

        executor = self.sym.simple_bind(ctx=mx.cpu(), **input_shapes)
        for key in executor.arg_dict.keys():
            if key in self.arg_params:
                self.arg_params[key].copyto(executor.arg_dict[key])

        executor.forward(is_train=True, data=mx.nd.array(img))
        probs = executor.outputs[0].asnumpy()
        line = ''
        label = []
        for i in range(probs.shape[0]):
            # if i < batch_size:
            #     result =  np.argmax(probs[i][0:31])
            # # if i == 1:
            # #     result =  np.argmax(probs[i][41:65])+41
            #     result =  np.argmax(probs[i][31:65])+31

            result = np.argmax(probs[i][0:69])

            label.append(result)
            line += self.chars[result]
        return line, label

        # cv2.waitKey(0)

    def Accuracy(self,pre, real, num):
        d = 0
        # print (pre[1],real[1])
        for i in range(num):
            if pre[i] == real[i]:
                d += 1
        # print(d/num)
        return d / num

    def Apart(self,aim):
        x = [None] * self.length
        z = 0
        b = m
        for i in range(self.length):
            x[i] = aim[z:b]
            z = b
            b = b + m

        y = [None] * m
        for j in range(m):
            s = []
            for i in range(self.length):
                s.append(x[i][j])
            y[j] = s
        return y

    def Apart_name(self,aim):
        x = [None] * self.length
        z = 0
        b = m
        for i in range(self.length):
            x[i] = aim[z:b]
            z = b
            b = b + m

        y = [None] * m
        for j in range(m):
            s = ''
            for i in range(self.length):
                s += x[i][j]
                # s.append(x[i][j])
            y[j] = s
        return y

    def Transfer(self,file):
        img_all = [None] * len(name_all)
        for i in range(len(name_all)):
            # img = cv2.imread(file_path+name_all[i])
            img = cv2.imdecode(np.fromfile(self.filepath + name_all[i], dtype=np.uint8), -1)
            img = cv2.resize(img, (120, 30))
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img_all[i] = img

        s = np.array(img_all)
        return s

    def Run(self, filepath):  # a is the length of plate , b is file path
        # name_all, labe_all, Pname_T = select_all_T(file_path)
        self.filepath = filepath
        global name_all
        global labe_all
        global m
        global n
        name_all, labe_all = self.select_all()
        s = self.Transfer(name_all)
        pre_name, pre_label = self.TestRecognizeOne(s)

        m = len(name_all)
        n = len(pre_label)
        label_pre = self.Apart(pre_label)
        name_pre = self.Apart_name(pre_name)
        accurace = self.Accuracy(label_pre, labe_all, m)

        # print('predicton label:',y[0],'num of plate:',m)
        # print('real label:',labe_all[0])
        # print('predicted name:', name_pre)
        # print('real name:', name_all)
        # print('Accuracy is :', accurace)
        return name_pre, name_all

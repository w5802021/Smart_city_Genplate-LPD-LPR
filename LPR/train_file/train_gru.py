import os
import argparse
import pandas as pd
import cv2
import numpy as np
from keras import backend as K
# from keras.layers import Input, Activation, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Dropout
# from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import *
from keras.layers import *

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         '港', '学', '使', '警', '澳', '挂', '军', '北', '南', '广',
         '沈', '兰', '成', '济', '海', '民', '航', '空',
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)

def ctc_lambda_func(args):
    '''
    调用keras内部函数实现的batch训练的ctc损失
    '''
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, :, 0, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(width, num_channels):
    '''
    定义模型
    :param width:图像的宽度
    :param num_channels: 图像通道数
    :return:
    '''
    input_tensor = Input(name='the_input', shape=(width, 48, num_channels), dtype='float32')
    x = input_tensor
    base_conv = 32
    rnn_size = 256
    # 多卷积核提取特征
    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()
    # 把多维数据平坦化，简单就是说把多维数据转化为一维数据
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    # 隐藏层神经元有32个
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 2个GRU添加，然后在2个GRU合并，使用了双向循环神经网络，一正一反
    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])

    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
    x = concatenate([gru_2, gru_2b])
    # 防止过拟合
    x = Dropout(0.25)(x)
    # 输出层 激活函数是softmax
    # len(CHARS)+1 其中1代表不识别为字符的序列图像
    x = Dense(len(CHARS)+1, init='he_normal', activation='softmax')(x)

    y_pred = x
    return input_tensor, y_pred


def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        if c == 'O':
            c = '0'
        label[i] = CHARS_DICT[c]
    return label

def parse_line(line):
    parts = line.split(' ')
    filename = parts[0]
    label = encode_label(parts[1].strip().upper())
    return filename, label

class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, input_length, num_channels=3, label_len=7):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._input_len = input_length
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()


class TextImageGenerator:

    def __init__(self, img_dir, label_file, batch_size, img_size, input_length, num_channels=3, label_len=7):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._input_len = input_length
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()

    def init(self):
        self.labels = []
        with open(self._label_file,encoding='utf-8') as f:
            for line in f:
                filename, label = parse_line(line)
                self.filenames.append(filename)
                self.labels.append(label)
                self._num_examples += 1

        # self.labels = [i.astype(np.int) for i in self.labels]

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = [self.labels[i] for i in perm]
            # self._labels = self.labels[perm]

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end >= self._num_examples:
            self._next_index = 0
            self._num_epoches += 1
            end = self._num_examples
            batch_size = self._num_examples - start
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])
        for j, i in enumerate(range(start, end)):
            try:
                fname = self._filenames[i]
                img = cv2.imdecode(np.fromfile(os.path.join(self._img_dir, fname), dtype=np.uint8), cv2.IMREAD_COLOR)
                # img = cv2.imread(os.path.join(self._img_dir, fname))
                img = cv2.resize(img,(164,48))
                images[j] = img
            except:
                print(self._filenames[i])
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end]
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        input_length[:] = self._input_len
        label_length[:] = self._label_len
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': np.array(images),
                  'the_labels': np.array(labels),
                  'input_length': input_length,
                  'label_length': label_length,}
        return inputs, outputs

    def get_data(self):
        '''
        训练数据生成器
        :return:每个batch的数据量
        '''
        while True:
            yield self.next_batch()

def train(c,log,label_len,img_size,num_channels,pre,ti,tl,b,vi,vl,n,start_epoch):
    '''
    训练模型
    :param args:
    :return:
    '''

    ckpt_dir = os.path.dirname(c)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if log != '' and not os.path.isdir(log):
        os.makedirs(log)
    label_len = label_len
    # 建立cnn模型
    input_tensor, y_pred = build_model(img_size[0], num_channels)
    # base_model = Model(input=input_tensor, output=y_pred)
    base_model = Model(inputs=input_tensor, outputs=y_pred)
    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')
    pred_length = int(y_pred.shape[1])
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    adam = Adam(lr=0.01, decay=1e-6)

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    if pre != '':
        model.load_weights(pre)

    train_gen = TextImageGenerator(img_dir=ti,
                                 label_file=tl,
                                 batch_size=b,
                                 img_size=img_size,
                                 input_length=pred_length,
                                 num_channels=num_channels,
                                 label_len=label_len)

    val_gen = TextImageGenerator(img_dir=vi,
                                 label_file=vl,
                                 batch_size=b,
                                 img_size=img_size,
                                 input_length=pred_length,
                                 num_channels=num_channels,
                                 label_len=label_len)

    checkpoints_cb = ModelCheckpoint(c, period=1)
    cbs = [checkpoints_cb]

    if log != '':
        tfboard_cb = TensorBoard(log_dir=log, write_images=True)
        cbs.append(tfboard_cb)

    model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=(train_gen._num_examples+train_gen._batch_size-1) // train_gen._batch_size,
                        epochs=n,
                        validation_data=val_gen.get_data(),
                        validation_steps=(val_gen._num_examples+val_gen._batch_size-1) // val_gen._batch_size,
                        callbacks=cbs,
                        initial_epoch=start_epoch)
    return base_model

def export(base_model,m):
    """
    输出模型的hdfs文件
    """
    model = base_model
    model.save(m)
    print('model saved to {}'.format(m))

def main ():
    # ps = argparse.ArgumentParser()
    # ps.add_argument('-num_channels', type=int, help='number of channels of the image', default=3)
    # subparsers = ps.add_subparsers()
    #
    # # Parser for arguments to train the model
    # parser_train = subparsers.add_parser('train', help='train the model')
    # parser_train.add_argument('-ti', help='训练图片目录', required=True)
    # parser_train.add_argument('-tl', help='训练标签文件', required=True)
    # parser_train.add_argument('-vi', help='验证图片目录', required=True)
    # parser_train.add_argument('-vl', help='验证标签文件', required=True)
    # parser_train.add_argument('-b', type=int, help='batch size', required=True)
    # parser_train.add_argument('-img-size', type=int, nargs=2, help='训练图片宽和高', required=True)
    # parser_train.add_argument('-pre', help='pre trained weight file', default='')
    # parser_train.add_argument('-start-epoch', type=int, default=0)
    # parser_train.add_argument('-n', type=int, help='number of epochs', required=True)
    # parser_train.add_argument('-label-len', type=int, help='标签长度', default=7)
    # parser_train.add_argument('-c', help='checkpoints format string', required=True)
    # parser_train.add_argument('-log', help='tensorboard 日志目录, 默认为空', default='')
    # parser_train.set_defaults(func=train)
    #
    # # Argument parser of arguments to export the model
    # parser_export = subparsers.add_parser('export', help='将模型导出为hdf5文件')
    # parser_export.add_argument('-m', help='导出文件名(.h5)', required=True)
    # parser_export.set_defaults(func=export)
    # args = ps.parse_args()
    # args.func(args)

    base_model = train(c='E:/smart_city/crnn_ctc_ocr/LPR/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5',log='',label_len=7,
                      img_size=(164,48),num_channels=3,pre='',ti=r'E:\smart_city\plate',tl='train.txt',b=32,
                      vi=r'E:\smart_city\plate',vl='valid.txt',n=1,start_epoch=0)
    export(base_model,m='./model/lpr_gru1.h5')

if __name__ == '__main__':
    main()
#coding=utf-8
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
from math import *
from tqdm import tqdm

# font = ImageFont.truetype("Arial-Bold.ttf",14)

# index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
#          "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
#          "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
#          "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
#          "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
#          "W": 61, "X": 62, "Y": 63, "Z": 64}

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "使", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
          "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
          "Y", "Z", "港", "澳", "学", "警", '领']

index = {chars[i]:i for i in range(len(chars))}

def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)

    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50]
    adder = cv2.resize(adder, (50, 50))
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

def rot(img,angel,shape,max_angel):
    '''
    图像旋转增强
    :param img: 输入图像
    :param angel: 旋转角度
    :param shape: 图片的目标尺寸
    :param max_angel:
    :return:
    '''

    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])

    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0]         ,[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):

        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)

    return dst

def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def tfactor(img):
    '''
    图像hsv增强
    :param img:
    :return:
    '''
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def random_envirment(img,data_set):
    '''

    :param img:
    :param data_set:
    :return:
    '''
    index=r(len(data_set))
    env = cv2.imread(data_set[index])

    env = cv2.resize(env,(img.shape[1],img.shape[0]))

    bak = (img==0)
    bak = bak.astype(np.uint8)*255
    inv = cv2.bitwise_and(bak,env)
    img = cv2.bitwise_or(inv,img)
    return img

def GenCh(f,val):
    img=Image.new("RGB", (45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3),val,(0,0,0),font=f)
    img =  img.resize((23,70))
    A = np.array(img)

    return A

def GenChSpecial(f,val):
    img=Image.new("RGB", (44,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3),val,(0,0,0),font=f)
    img =  img.resize((22,70))
    A = np.array(img)
    return A

def GenCh1(f,val):
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val,(0,0,0),font=f)
    A = np.array(img)
    return A

def GenChSpecial1(f,val):
    img=Image.new("RGB", (26,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val,(0,0,0),font=f)
    img = img.resize((22, 70))
    A = np.array(img)
    return A

def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
    diff = 255-single.max()
    noise = np.random.normal(0,1+r(6),single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0])
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1])
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2])
    return img

class GenPlate:

    def __init__(self,fontCh,fontEng,NoPlates):
        self.fontC1 =  ImageFont.truetype(fontCh,45,0)
        self.fontC2 = ImageFont.truetype(fontCh, 40, 0)
        self.fontE =  ImageFont.truetype(fontEng,58,0)
        self.img=np.array(Image.new("RGB", (226,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("./images/template.bmp"),(226,70))
        self.bg_black = cv2.resize(cv2.imread("./images/3.bmp"), (226, 70))
        self.bg_yellow = cv2.resize(cv2.imread("./images/5.bmp"), (226, 70))
        self.bg_green = cv2.resize(cv2.imread("./images/18.png"), (226, 70))
        self.smu = cv2.imread("./images/smu2.jpg")
        self.noplates_path = []
        for parent,parent_folder,filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent+"/"+filename
                self.noplates_path.append(path)

    def draw(self,val):
        offset= 2
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC1,val[0])
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1])

        for i in range(5):
            base = offset+8+23+6+23+17 +i*23 + i*6
            self.img[0:70, base:base+23]= GenCh1(self.fontE,val[i+2])
        return self.img

    def draw_special(self,val):
        offset= 2
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC1,val[0])
        self.img[0:70,offset+8+23+6:offset+8++6+23+23]= GenCh1(self.fontE,val[1])

        for i in range(5):
            base = offset+8+23+6+23+17 +i*23 + i*4
            if i == 4:
                self.img[0:70, base:base + 23 ] = GenChSpecial(self.fontC1, val[i + 2])
            else:
                self.img[0:70, base:base + 23] = GenCh1(self.fontE, val[i + 2])
        # cv2.imshow('aa',self.img)
        # cv2.waitKey(0)

        return self.img

    def draw_special2(self,val):
        offset= 2
        self.img[0:70,offset+8:offset+8+22]= GenChSpecial(self.fontC1,val[0])
        self.img[0:70,offset+8+22+6:offset+8++6+22+22]= GenChSpecial1(self.fontE,val[1])

        for i in range(6):
            base = offset+8+22+6+22+12 +i*22 + i*3
            self.img[0:70, base:base + 22] = GenChSpecial1(self.fontE, val[i + 2])

        # cv2.imshow('aa',self.img)
        # cv2.waitKey(0)
        return self.img

    def generate(self,text):
        # a = len(text)
        if len(text) == 7:
            fg = self.draw_special(text)
            fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_or(fg,self.bg)
            com = rot(com,r(60)-30,com.shape,30)
            com = rotRandrom(com,10,(com.shape[1],com.shape[0]))
            #com = AddSmudginess(com,self.smu)
            com = tfactor(com)
            com = random_envirment(com,self.noplates_path)
            com = AddGauss(com, 1+r(4))
            com = addNoise(com)
            return com

    def generateSpecial(self,text):
        '''
        产生特殊车牌
        :param text:
        :return:
        '''
        # if len(text) == 7:
        #     fg = self.draw_special(text)
        #     fg = cv2.bitwise_not(fg)
        #     com = cv2.bitwise_or(fg,self.bg_black)
        #
        #     com = rot(com,r(60)-30,com.shape,30)
        #     # com = rotRandrom(com,10,(com.shape[1],com.shape[0]))
        #     #com = AddSmudginess(com,self.smu)
        #     com = tfactor(com)
        #     # com = random_envirment(com,self.noplates_path)
        #     # com = AddGauss(com, 1+r(4))
        #     com = addNoise(com)
        #     cv2.imshow('aa', com)
        #     cv2.waitKey(0)

        # if len(text) == 7:
        #     fg = self.draw_special(text)
        #
        #     # fg = cv2.bitwise_not(fg)
        #     com = cv2.bitwise_and(fg, self.bg_yellow)
        #
        #     com = rot(com, r(60) - 30, com.shape, 30)
        #     # com = rotRandrom(com,10,(com.shape[1],com.shape[0]))
        #     # com = AddSmudginess(com,self.smu)
        #     com = tfactor(com)
        #     # com = random_envirment(com,self.noplates_path)
        #     # com = AddGauss(com, 1+r(4))
        #     com = addNoise(com)
        #     # cv2.imshow('aa', com)
        #     # cv2.waitKey(0)

        if len(text) == 8:
            fg = self.draw_special2(text)

            # fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_and(fg, self.bg_green)

            com = rot(com, r(60) - 30, com.shape, 30)
            # com = rotRandrom(com,10,(com.shape[1],com.shape[0]))
            # com = AddSmudginess(com,self.smu)
            com = tfactor(com)
            # com = random_envirment(com,self.noplates_path)
            # com = AddGauss(com, 1+r(4))
            com = addNoise(com)
            # cv2.imshow('aa', com)
            # cv2.waitKey(0)

        return com

    def genPlateString(self,pos,val):
        plateStr = ""
        box = [0,0,0,0,0,0,0]
        if(pos!=-1):
            box[pos]=1
        for unit,cpos in zip(box,range(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    plateStr += chars[r(33)]
                elif cpos == 1:
                    # 产生第2位的字母
                    plateStr += chars[42+r(24)]
                else:
                    # 产生数字和字母
                    plateStr += chars[31 + r(34)]
        return plateStr

    def genSpecialPlateString(self, pos, val):
        '''
        产生特殊车牌
        :param pos:
        :param val:
        :return:
        '''
        #港澳车牌
        # plateStr = ""
        # box = [0, 0, 0, 0, 0, 0, 0]
        # if (pos != -1):
        #     box[pos] = 1
        # for unit, cpos in zip(box, range(len(box))):
        #     if unit == 1:
        #         plateStr += val
        #     else:
        #         if cpos == 0:
        #             plateStr += '粤'
        #         elif cpos == 1:
        #             plateStr += 'Z'
        #         elif cpos == 6:
        #             plateStr += chars[66 + r(2)]
        #         else:
        #             plateStr += chars[32 + r(34)]
        #教练车
        # plateStr = ""
        # box = [0, 0, 0, 0, 0, 0, 0]
        # if (pos != -1):
        #     box[pos] = 1
        # for unit, cpos in zip(box, range(len(box))):
        #     if unit == 1:
        #         plateStr += val
        #     else:
        #         if cpos == 0:
        #             plateStr += chars[r(32)]
        #         elif cpos == 1:
        #             plateStr += chars[42 + r(24)]
        #         elif cpos == 6:
        #             plateStr += '学'
        #         else:
        #             plateStr += chars[32 + r(34)]

        plateStr = ""
        box = [0, 0, 0, 0, 0, 0, 0, 0]
        if (pos != -1):
            box[pos] = 1
        for unit, cpos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    plateStr += chars[r(32)]
                elif cpos == 1:
                    plateStr += chars[42+r(24)]
                else:
                    plateStr += chars[32 + r(34)]
        return plateStr

    def genBatch(self, batchSize,pos,charRange, outputPath,size):
        '''
        批量产生车牌并输出文件
        :param batchSize: 产生车牌数量
        :param pos:
        :param charRange:
        :param outputPath:
        :param size:
        :return:
        '''
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in tqdm(range(0, batchSize)):
            plateStr = G.genSpecialPlateString(-1,-1)
            img =  G.generateSpecial(plateStr)
            img = cv2.resize(img,size)
            filename = os.path.join(outputPath, str(i).zfill(4) + '.' + plateStr + ".jpg")
            # cv2.imwrite(filename, img)
            cv2.imencode('.jpg',img)[1].tofile(filename)

if __name__ == '__main__':
    G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")
    # G.genBatch(10000,2,range(31,65),"./plate_train",(272,72))
    G.genBatch(30000,2,range(31,65),"./plate_test",(136,36))



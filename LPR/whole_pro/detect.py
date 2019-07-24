import cv2
############################################################################
#   本文件主要是：从输入的原始图片中定位并提取出车牌
#   包含函数：def computeSafeRegion：
#             def cropped_from_image(image, rect):
#             def detectPlateRough(image_gray, resize_h = 720, en_scale =1.08, top_bottom_padding_rate = 0.05):
#   其中第三个函数是主要函数，前两个被调用
############################################################################

watch_cascade = cv2.CascadeClassifier('./model/cascade.xml')     # 加载级联分类器  http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/user_guide/ug_traincascade.html#id11
                                                                    # http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html#cascade-classifier
def computeSafeRegion(shape,bounding_rect):
    '''
    功能：计算图像切分安全区域
    :param shape:
    :param bounding_rect:
    :return:
    '''
    top = bounding_rect[1]                                         # y
    bottom = bounding_rect[1] + bounding_rect[3]                  # y +  h
    left = bounding_rect[0]                                        # x
    right = bounding_rect[0] + bounding_rect[2]                    # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

    # print "computeSateRegion input shape",shape
    if top < min_top:
        top = min_top
        # print "tap top 0"
    if left < min_left:
        left = min_left
        # print "tap left 0"

    if bottom > max_bottom:
        bottom = max_bottom
        # print "tap max_bottom max"
    if right > max_right:
        right = max_right
        # print "tap max_right max"

    # print "corr",left,top,right,bottom
    return [left, top, right-left, bottom-top]


def cropped_from_image(image, rect):
    '''
    功能：从图像计算安全的剪切区域
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h = computeSafeRegion(image.shape, rect)
    return image[y:y+h, x:x+w]

def detectPlateRough(image_gray, resize_h = 720, en_scale =1.08, top_bottom_padding_rate = 0.00):
    '''
    功能：车牌粗定位
    :param image_gray: 原始图像灰度图
    :param resize_h: 图像尺寸压缩后的高度
    :param en_scale: 压缩图像的长宽比例
    :param top_bottom_padding_rate: 图像上下区域阈值剪切比例
    :return:粗定位后的车牌
    '''
    # print(image_gray.shape)

    if top_bottom_padding_rate > 0.2:
        print("error:top_bottom_padding_rate > 0.2:", top_bottom_padding_rate)
        exit(1)

    height = image_gray.shape[0]
    padding = int(height*top_bottom_padding_rate)
    # 灰度图像的长宽比
    scale = image_gray.shape[1]/float(image_gray.shape[0])

    image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
    # 裁剪掉padding的上下图片区域
    image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
    image_gray = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
    # 使用训练好的cascade.xml的模型进行车牌定位，返回检测到的可能的车牌区域
    # void detectMultiScale(const Mat& image, CV_OUT vector<Rect>& objects, double scaleFactor = 1.1,
    #                        int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())
    # scaleFactor(en_scale):每次缩小图像的比例；minNeighbors：匹配成功所需要的周围矩阵框的数目
    # scaleFactor越小，金字塔层级越多，计算慢，更准确
    # 输出示例：[[390 255 174  44]]
    watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9), maxSize=(36*40, 9*40))
    cropped_images = []
    # for (x, y, w, h) in watches:
    #     cropped_origin = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))      # 输入的是彩色图
    #     x -= w * 0.14
    #     w += w * 0.28
    #     y -= h * 0.6
    #     h += h * 1.1
    #
    #     cropped = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))             # 输入的是彩色图
    #     这里使得返回的值并不是一张图
    #     cropped_images.append([cropped, [x, y+padding, w, h], cropped_origin])
    for j, (x, y, w, h) in enumerate(watches):
        # 仅取cascade.xml模型识别出的车牌区域置信度最高的图片
        if j == 0:
            cropped_origin = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
            # 对在安全区域的原始图像进行剪切并对上下左右边界进行扩充，得到最后的粗定位的车牌区域
            x -= w * 0.2   # 0.14
            w += w * 0.4   # 0.28
            y -= h * 0.05  # 0.6
            h += h * 0.1   # 1.1
            cropped = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
            cropped_images.append([cropped, [x, y+padding, w, h], cropped_origin])
            break
    # 返回cropped：最终粗定位的图像 [x, y+padding, w, h]：粗定位图像在原图像的位置 cropped_origin：未经过切割的图像
    return cropped_images

import cv2
import os
import hashlib

# def verticalMappingToFolder(image, path):
#     name = hashlib.md5(image.data).hexdigest()[:8]
#     print(name)
#     cv2.imwrite("./cache/finemapping/"+name+".png", image)

##############################################################
#                 本函数保存完整车牌
##############################################################
def verticalMappingToFolder(image, path, name = "", color_type = 0):
    print("name:", name)
    temp = path.split('/')
    path = "./OutDataset/完整车牌/" + temp[len(temp)-2] + "/" + temp[len(temp)-1]
    if not os.path.exists(path):
        os.makedirs(path)

    if (color_type > 0) and (color_type < 5):
        image = cv2.bitwise_not(image)

    cv2.imencode('.jpg', image)[1].tofile(path + "/" + name)

##############################################################
#                 本函数保存分割车牌
##############################################################
def verticalMappingToFolder_segmentation(img_array, path, name = ""):
    print("name:", name)
    temp = path.split('/')
    path = "./OutDataset/分割车牌/" + temp[len(temp)-2] + "/" + temp[len(temp)-1] + "/" + name
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(7):
        cv2.imencode('.jpg', img_array[i])[1].tofile(path + "/" + str(i) + "-" + name)


##############################################################
#                 本函数保存分割车牌训练集
##############################################################
Perfor_segment_plate = ["粤A6503H", "粤A177US", "粤A3659D", "粤A0RE33", "粤A4102J", "粤A5RW15", "粤A9CE55", "粤AC777X", "粤A7AB78", "粤A8716Z",
                        "粤AY658X", "粤A8KS92", "粤A0TH69", "粤A5HR14", "粤AL372R", "粤A0GM68", "粤A973FS", "粤A6860U", "粤A205ZE", "粤A2KC31"]

def Ability_save_segmentation_train(img_array, path, name = ""):
    print("save_train name:", name)
    for i in range(7):
        if name[i] == ".":         # 针对名称只有6位字符的车牌，如：使00230.jpg
            break
        if name[i] == "I":
            path = "./OutDataset/分割车牌训练集/" + "1"
        else:
            path = "./OutDataset/分割车牌训练集/" + name[i]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imencode('.jpg', img_array[i])[1].tofile(path + "/" + str(i) + "-" + name)

def Perform_save_segmentation_train(img_array, path, name = ""):
    print("name:", name)
    name_real = Perfor_segment_plate[int(name[len(name)-6:len(name)-4])-1]
    for i in range(7):
        path = "./OutDataset/分割车牌训练集/" + name_real[i]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imencode('.jpg', img_array[i])[1].tofile(path + "/" + str(i) + "-" + name)
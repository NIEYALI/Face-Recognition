from __future__ import print_function
"""
该文件将所有数据文件进行处理
"""

import os
import cv2
import numpy as np
Image_size=64



def getPaddingSize(image,height=Image_size,width=Image_size):
    #初始化人脸坐标
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    longest = max(h, w)
    if h<longest:
        dh=longest-h
        top=dh//2
        bottom=dh-top
    elif w<longest:
        dw=longest-w
        left=dw//2
        right=dw-left
    else:
        pass
    # BGR颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant,(height,width))

images=[]
labels=[]
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = getPaddingSize(image, Image_size, Image_size)

                # 放开这个代码，可以看到resize_image()函数的实际调用效果
                # cv2.imwrite('1.jpg', image)
                images.append(image)
                labels.append(path_name)
    return images,labels




def load_dataset(path_name):
    """

    :param path_name: 查询所有文件
    :return:
    """
    images, labels = read_path(path_name)
    images = np.array(images)
    print(images.shape)
    newlabels = []

    for label in labels:
        if label.endswith('face_0'):
            newlabels.append(0)
        elif label.endswith('face_1'):
            newlabels.append(1)
        elif label.endswith('face_2'):
            newlabels.append(2)
        elif label.endswith('face_3'):
            newlabels.append(3)
        elif label.endswith('face_4'):
            newlabels.append(4)
        elif label.endswith('face_5'):
            newlabels.append(5)
        elif label.endswith('face_6'):
            newlabels.append(6)
        elif label.endswith('face_7'):
            newlabels.append(7)
        elif label.endswith('face_8'):
            newlabels.append(8)
        elif label.endswith('face_9'):
            newlabels.append(9)
        elif label.endswith('face_10'):
            newlabels.append(10)
        elif label.endswith('face_11'):
            newlabels.append(11)
        elif label.endswith('face_12'):
            newlabels.append(12)
        elif label.endswith('face_13'):
            newlabels.append(13)
        elif label.endswith('face_14'):
            newlabels.append(14)
        elif label.endswith('face_15'):
            newlabels.append(15)
        elif label.endswith('face_16'):
            newlabels.append(16)
        elif label.endswith('face_17'):
            newlabels.append(17)
        elif label.endswith('face_18'):
            newlabels.append(18)
        elif label.endswith('face_19'):
            newlabels.append(19)
        elif label.endswith('face_20'):
            newlabels.append(20)
        elif label.endswith('face_21'):
            newlabels.append(21)
        elif label.endswith('face_22'):
            newlabels.append(22)
        elif label.endswith('face_23'):
            newlabels.append(23)
        elif label.endswith('face_24'):
            newlabels.append(24)
        elif label.endswith('face_25'):
            newlabels.append(25)
        elif label.endswith('face_26'):
            newlabels.append(26)
        elif label.endswith('face_27'):
            newlabels.append(27)
        elif label.endswith('face_28'):
            newlabels.append(28)
        elif label.endswith('face_29'):
            newlabels.append(29)
        elif label.endswith('face_30'):
            newlabels.append(30)
        elif label.endswith('face_31'):
            newlabels.append(31)
        elif label.endswith('face_32'):
            newlabels.append(32)
        elif label.endswith('face_33'):
            newlabels.append(33)
        elif label.endswith('face_34'):
            newlabels.append(34)
        elif label.endswith('face_35'):
            newlabels.append(35)
        elif label.endswith('face_36'):
            newlabels.append(36)
        elif label.endswith('face_37'):
            newlabels.append(37)
        elif label.endswith('face_38'):
            newlabels.append(38)
        elif label.endswith('face_39'):
            newlabels.append(39)
        elif label.endswith('face_40'):
            newlabels.append(40)
        elif label.endswith('face_41'):
            newlabels.append(41)

        else:
            newlabels.append(42)
    labels = np.array(newlabels)
    print(labels)

    return images, labels

if __name__=='__main__':
    path_name='/home/lyl/Documents/allcode_Model/CNN_Face/dataset/Myface/'
    load_dataset(path_name)

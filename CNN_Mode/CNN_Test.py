# coding=UTF-8
from __future__ import print_function
"""
文件数据的导入和创建
"""
#camera_idx=192.168.1.64
import cv2
from CNN_Face import Model
import dlib
Image_size=64


def Rec_Face(camera_idx,file_path):
    # 加载模型
    model = Model()
    model.load_model(file_path)
    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)
    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(camera_idx)
    detector = dlib.get_frontal_face_detector()
    # 人脸识别分类器本地存储路径
    # 循环检测识别人脸
    while True:
        _, image = cap.read()  # 读取一帧视频
        # 图像灰化，降低计算复杂度
        #photo_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        photo_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用人脸识别分类器，读入分类器
        dst = detector(photo_gray, 1)
        # 利用分类器识别出哪个区域为人脸
        if len(dst) > 0:
            for i, d in enumerate(dst):
                top = d.top() if d.top() > 0 else 0
                bottom= d.bottom() if d.bottom() > 0 else 0
                left= d.left() if d.left() > 0 else 0
                right= d.right() if d.right() > 0 else 0
                face = image[top:bottom, left:right]
                faceID = model.face_predict(face)

                if faceID == 0:
                    tag = '我'
                elif faceID == 1:
                    tag = 'face_1'
                elif faceID == 2:
                    tag = 'face_2'
                elif faceID == 3:
                    tag = 'face_3'
                elif faceID == 4:
                    tag = 'face_4'
                elif faceID == 5:
                    tag = 'face_5'
                elif faceID == 6:
                    tag = 'face_6'
                elif faceID == 7:
                    tag = 'face_7'
                elif faceID == 8:
                    tag = 'face_8'
                elif faceID == 9:
                    tag = 'face_9'
                elif faceID == 10:
                    tag = 'face_10'
                elif faceID == 11:
                    tag = 'face_11'
                elif faceID == 12:
                    tag = 'face_12'
                elif faceID == 13:
                    tag = 'morface'
                else:
                    pass
                face = cv2.resize(face, (Image_size, Image_size))
                # print('Is this my face? %s' % Main_model(face))
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
                # 文字提示是谁, 坐标 字体 字号 颜色 字的线宽
                cv2.putText(image, tag, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Surprise", image)
        k = cv2.waitKey(10)
        # 按q退出窗口，注意opencv 不支持窗口的关闭按钮关闭窗口
        if k & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    return camera_idx,file_path

if __name__=='__main__':
    #camera_idx = 'rstp://admin:zm891127@192.168.1.64:554/11'
    #camera_idx = 'http://admin:zm891127@192.168.1.64:80/11'
    #camera_idx = 'rstp://admin:zm891127@192.168.1.64:554/h264/ch1/main/av_stream'
    camera_idx = 0
    file_path = '/home/nyl/Documents/CODE_FACE/Keras_Code/CNN_Mode/models/CNN.face.model.h5'
    Rec_Face(camera_idx, file_path)

from __future__ import print_function
"""
导入数据包之类的东西
"""

import sys
import os
import dlib
import cv2


class DectMorFace:

    def __int__(self):
        self.DectMorFace=None

    """
    数据处理
    """
    @staticmethod
    def ReadImage(Image_size,input_image,output_image):

        if not os.path.exists(input_image):
            print("文件不存在，需要重新导入输入数据")
        else:
            print("文件存在，继续以下操作！！！")
            try:

                print(';;;;;;')
                index = 1
                for dirpath,dirnames,filenames in os.walk(input_image):

                    for filename in filenames:
                        if filename.endswith('.jpg'):
                            print('Being processed picture %s' % index)
                            filename=dirpath+'/'+filename
                            image=cv2.imread(filename)
                            frame_new=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                            detector=dlib.get_frontal_face_detector()
                            dets=detector(frame_new,1)
                            for i ,face in enumerate(dets):
                                """
                                图像中人脸位置的检测和表示
                                """
                                top = face.top() if face.top() > 0 else 0
                                bottom = face.bottom() if face.bottom() > 0 else 0
                                left = face.left() if face.left() > 0 else 0
                                right = face.right() if face.right() > 0 else 0
                                # 保存脸部图片
                                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
                                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
                                    i, face.left(), face.top(), face.right(), face.bottom()))
                                image_new=image[top:bottom, left:right]
                                image_new = cv2.resize(image_new, (Image_size, Image_size))
                                cv2.imshow('image', image_new)
                                cv2.imwrite(output_image + '/' + 'Face_'+str(index) + '.jpg', image_new)
                                index += 1
                            key = cv2.waitKey(30) & 0xff
                            if key == 27:
                                sys.exit(0)
            except:
                pass
        return input_image,output_image








if __name__=='__main__':

    Image_size=64
    input_image = '/home/lyl/Documents/CODE_FACE/Keras_Code/dataset/input_image'
    output_image = '/home/lyl/Documents/CODE_FACE/Keras_Code/dataset/Myface/morface/'
    dectmorface=DectMorFace()
    dectmorface.ReadImage(Image_size,input_image,output_image)
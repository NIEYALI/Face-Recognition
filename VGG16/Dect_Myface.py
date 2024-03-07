from __future__ import print_function
"""
导入数据包
"""
import os
import cv2
import dlib


n = 0
class DectFace:

    def __int__(self):
        self.DectFace=None

    @staticmethod

    def DectFace(Image_size,Win_name,face_num_max,camera_idx,path_name):
        if not os.path.exists(path_name):
            print('Error: "', path_name, '" 文件不存在！！')
        else:
            print('继续下面测试')
            """
            创建全局不变量
            """
            global n
            try:
                for dirpath,dirnames,filenames in os.walk(path_name):
                    for filepath in dirnames:
                        sub_path=os.path.join(dirpath,filepath)
                        n+=1
                        print('No',n,' ',sub_path)
            except:
                pass
        detector = dlib.get_frontal_face_detector()
        cv2.namedWindow(Win_name)
        camera = cv2.VideoCapture(camera_idx)

        if not camera.isOpened():
            print("cannot open camear")
            exit(0)
        else:
            print("摄像头链接继续测试！")

        for t in range(n,n+1):
            filename=path_name+'face_'+str(t)
            os.mkdir(filename)
            t=t+1
            print('filename=:',filename)
            print('t=:',t)
            # classes=t
            # print(classes)

        num=0
        while True:
            if (num <= face_num_max):
                #print("11111")
                ret, frame = camera.read()
                if not ret:
                    break
                frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 检测脸部
                dets = detector(frame_new, 1)
                print("Number of faces detected: {}".format(len(dets)))
                # 查找脸部位置
                for i, face in enumerate(dets):
                    left = face.left()
                    top = face.top()
                    right = face.right()
                    bottom = face.bottom()
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

                    # 保存脸部图片
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(
                        i, face.left(), face.top(), face.right(), face.bottom()))
                    # 绘制脸部位置

                    # img1=frame[face.top():face.bottom(),face.left():face.right()]
                    img1 = frame[top:bottom, left:right]
                    img1 = cv2.resize(img1, (Image_size, Image_size))
                    num = num + 1

                    cv2.imwrite(filename + '/' +'FACE_'+str(num) + '.jpg', img1)
                cv2.imshow(Win_name, frame)
                key = cv2.waitKey(10)
                if key & 0xFF == ord('q'):
                    break
            else:
                print('完成！')
                break
        camera.release()
        cv2.destroyAllWindows()

        return Image_size,Win_name,face_num_max,camera_idx,path_name


if __name__=='__main__':

    path_name = '/home/lyl/Documents/CODE_FACE/Keras_Code/dataset/Myface/'
    Image_size = 64
    camera_idx=0

    face_num_max=100

    Win_name='Face'
    dectface=DectFace()
    dectface.DectFace(Image_size,Win_name,face_num_max,camera_idx,path_name)


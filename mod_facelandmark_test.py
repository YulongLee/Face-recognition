# coding: utf-8
#   模型测试
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014

import os
import cv2
import dlib
import glob

# 01-路径设置
current_path = os.getcwd()
faces_path = current_path + '/data/landmark'

# 模型加载
predictor = dlib.shape_predictor("model/predictor.dat")
detector = dlib.get_frontal_face_detector()
print("Showing detections and predictions on the images in the faces folder...")
for f in glob.glob(os.path.join(faces_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 02-人脸区域检测
    dets = detector(img2, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))
        # 02-人脸区域绘制
        # left = face.left()
        # top = face.top()
        # right = face.right()
        # bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        # cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        # cv2.imshow(f, img)

        # 03-特征点检测
        shape = predictor(img, face)
        # print(shape)
        # print(shape.num_parts)
        for index, pt in enumerate(shape.parts()):
            print('Part {}: {}'.format(index, pt))
            pt_pos = (pt.x, pt.y)
            # 04-特征点绘制
            cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
        # print(type(pt))
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
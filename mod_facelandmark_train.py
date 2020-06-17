# coding: utf-8
#   人脸特征点检测模型训练
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014

import os
import cv2
import dlib
import glob

# 00-样本路径
current_path = os.getcwd()
faces_path = current_path + '/data/landmark'

# 01-参数设置
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

# 02-导入打好了标签的xml文件
training_xml_path = os.path.join(faces_path, "training_with_face_landmarks.xml")
# 03-进行训练，训练好的模型将保存为predictor.dat
print(training_xml_path)
dlib.train_shape_predictor(training_xml_path, "model/predictor.dat", options)
# 04-打印在训练集中的准确率
print
"\nTraining accuracy:{0}".format(dlib.test_shape_predictor(training_xml_path, "model/predictor.dat"))

# 05-导入测试集的xml文件
testing_xml_path = os.path.join(faces_path, "testing_with_face_landmarks.xml")
# 打印在测试集中的准确率
print
"\Testing accuracy:{0}".format(dlib.test_shape_predictor(testing_xml_path, "model/predictor.dat"))

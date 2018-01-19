#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
# from model.config import cfg
# from model.test import im_detect
# from model.nms_wrapper import nms
#
# from utils.timer import Timer
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import os, cv2
# import argparse
#
# from nets.vgg16 import vgg16
# from nets.resnet_v1 import resnetv1

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
#
# NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
# DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


import numpy.random as npr
import numpy as np
import pickle
import os

org_file = r'D:\用户目录\下载\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt'
out_path = r'E:\pascal_voc\trainset\train2000\trainval.txt'
out_val_path = r'E:\pascal_voc\trainset\train2000\test.txt'
active_pkl = r'E:\tffrcnn_results\train_1000_results\test\voc_2007_test\default\res101_faster_rcnn_iter_30000\active.pkl'
test_file = r'D:\用户目录\下载\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\ImageSets\Main\test.txt'
# generate training set
# with open(org_file,'r') as f:
#     image_index = [x.strip() for x in f.readlines()]
#
# image_index_tonp = np.array(image_index).astype(int)
# index = npr.choice(image_index_tonp, size=1000, replace=False)
# with open(out_path, 'w') as f:
#     for i in index:
#         f.write('%06d'%i+'\n')

# with open(r'E:\tffrcnn_results\all_images_70000_results\output\res101\voc_2007_test\default\res101_faster_rcnn_iter_70000\detections.pkl','rb') as f:
#     out = pickle.load(f)

# generate testing set
def generate_testset(selected_path, allset_path, outset_path):
    with open(allset_path,'r') as f:
        image_index = [x.strip() for x in f.readlines()]

    with open(selected_path, 'r') as f:
        select_index = [x.strip() for x in f.readlines()]

    test_index = [ x for x in image_index if x not in select_index]
    with open(outset_path, 'w') as f:
        for i in test_index:
            f.write(i + '\n')


# active learning using entropy
high_confidence = 0.9
low_confidence = 0.05
size = 1000
def entropy(scores):
    ignor_count = 0
    entropy = 0
    result = []
    for i in range(len(scores)):
        if scores[i, -1] < low_confidence:
            ignor_count += 1
        else:
            entropy += -(scores[i, -1])*np.log(scores[i, -1])
    result.append(scores[0,0])
    result.append(ignor_count)
    result.append(entropy)

    return result

entropy_results = []
with open(active_pkl, 'rb') as f:
    results = pickle.load(f)
for i in range(len(results)):
    if i == 0:
        entropy_results = np.array(entropy(results[i]))
    else:
        entropy_results = np.vstack((entropy_results, np.array(entropy(results[i]))))

keep_index = np.argsort(entropy_results[:, -1])[::-1]
keep_index = keep_index[0:size]
final_results = entropy_results[keep_index, :]
save_path =r'E:\pascal_voc\trainset\train2000\trainval.txt'
with open(save_path, 'w') as f:
    for i in range(len(final_results)):
        f.write('%06d'%(int(final_results[i][0]),)+'\n')

form_test_path = r'E:\pascal_voc\trainset\train1000\test.txt'
current_test_path = r'E:\pascal_voc\trainset\train2000\test.txt'
generate_testset(save_path, form_test_path, current_test_path)



import xml.etree.ElementTree as ET
## count the number of objects
# annotation_path = r'D:\用户目录\下载\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations'
# annotation_test_path = r'D:\用户目录\下载\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Annotations'
#
# def parse_rec(filename):
#   """ Parse a PASCAL VOC xml file """
#   tree = ET.parse(filename)
#   objects = tree.findall('object')
#
#   return len(objects)
#
#
# with open(org_file,'r') as f:
#     image_index = [x.strip() for x in f.readlines()]
#
# count =[0 for i in range(50)]
# flag = 0
# for x in image_index:
#     xml_path = os.path.join(annotation_path, x+'.xml')
#     number = parse_rec(xml_path)
#     count[number] += 1
    # if number > 40:
    #     flag +=1
    # if flag > 0:
    #     print(x)
    #     break
#print(count)
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:  ##__C.TEST.SCALES = (600,)
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:  ##__C.TEST.MAX_SIZE = 1000
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)                     ## blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])
  
  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  ## for active learning
  Con_thresh = 0.5
  all_image = [[] for i in range(num_images)]
  ob_per_image = 3
  feature = {}
  ## upper for active learning
  pkl_filepath = '/home/xf/tf-faster-rcnn/output/res101/voc_2007_test/default/res101_faster_rcnn_iter_30000/detectioins.pkl'
  if os.path.isfile(pkl_filepath):
    with open(pkl_filepath, 'rb') as fid:
      all_boxes = pickle.load(fid)
  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    _t['im_detect'].tic()
    scores, boxes = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    #feature[int(imdb.image_index[i])] = net.extract_head(sess, im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))) # key in the feature is int type

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)  # 0.3
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time))

    det1 = []
    count = 0
    ## upper for active learning
##for active learning voc2007 has 9963 images and 24640 (29952)annotated objects so,approximately 2.47 objects per images
    # count = 0
    # det1 = []
    # for k in range(1, imdb.number_classes):
    #   det = all_boxes[k][i]
    #   index = np.where[det[:, -1] >= Con_thresh][0]
    #   if len(index) == 0:
    #      continue
    #   if count == 0:
    #     det = det(index)
    #     det = np.hstack((np.ones((len(index), 1))*k , det))
    #     det1 = det
    #   else:
    #     det = det(index)
    #     det = np.hstack((np.ones((len(index), 1)) * k, det))
    #     det1 = np.vstack(det1, det[index])
    #   count += len(index)
    #
    # if count > ob_per_image:
    #   keep = np.argsort(det1[:, -1])[::-1]
    #   det1 = det1[keep, :]
    if ob_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                                for j in range(1, imdb.num_classes)])
      if len(image_scores) > ob_per_image:
        image_thresh = np.sort(image_scores)[-ob_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
          if len(keep) == 0:
            continue
          if count == 0:
            det = all_boxes[j][i]
            det = np.hstack((np.ones((len(keep), 1)) * j, det))
            det = np.hstack((np.ones((len(keep), 1)) * int(imdb.image_index[i]), det))
            det1 = det
          else:
            det = all_boxes[j][i]
            det = det[keep]
            det = np.hstack((np.ones((len(keep), 1)) * j, det))
            det = np.hstack((np.ones((len(keep), 1)) * int(imdb.image_index[i]), det))
            det1 = np.vstack((det1, det[keep]))
          count += len(keep)
    if count > ob_per_image:
        temp_index = np.argsort(det1[:, -1])[::-1]
        save_index = temp_index[0:ob_per_image]
        det1 = det1[save_index, :]
        count = det1.shape[0]
    assert count == ob_per_image
    all_image[i] = det1

  active_file = os.path.join(output_dir, 'active.pkl')
  with open(active_file, 'wb') as f:
    pickle.dump(all_image, f, pickle.HIGHEST_PROTOCOL)

  # feature_file = os.path.join(output_dir, 'feature.pkl')
  # with open(feature_file, 'wb') as f:
  #   pickle.dump(feature, f, pickle.HIGHEST_PROTOCOL)

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)


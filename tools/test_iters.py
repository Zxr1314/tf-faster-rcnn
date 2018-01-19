#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

import pickle
import os
import glob

# nfile = '/home/xf/tf-faster-rcnn/output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_30000.pkl'
# with open(nfile, 'rb') as fid:
#     st0 = pickle.load(fid)
#     cur = pickle.load(fid)
#     perm = pickle.load(fid)
#     cur_val = pickle.load(fid)
#     perm_val = pickle.load(fid)
#     last_snapshot_iter = pickle.load(fid)
#
# print(last_snapshot_iter)

class test():
    def __init__(self):
        self.output_dir = r'E:\tffrcnn_results\all_images_70000_results\output\res101\voc_2007_trainval\default'
        self.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
        self.STEPSIZE = [80000]
    def find_previous(self):
        sfiles = os.path.join(self.output_dir,
                              self.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')  ##__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in self.STEPSIZE:  ##__C.TRAIN.STEPSIZE = [30000]
            redfiles.append(os.path.join(self.output_dir,
                                         self.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, self.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

if __name__ == '__main__':
    a = test()
    l, n, s = a.find_previous()
    print(l, n ,s)

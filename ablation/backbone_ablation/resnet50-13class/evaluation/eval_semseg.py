# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import glob
import json
import numpy as np
import torch
from PIL import Image
import pdb

VOC_CATEGORY_NAMES = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


NYU_CATEGORY_NAMES = ['bed', 'books', 'ceiling', 'chair', 'floor', 'furniture', 'object',
                       'picture', 'sofa', 'table', 'tv', 'wall', 'window']

class SemsegMeter(object):
    def __init__(self, database, ignore_idx=255):
        ''' "marco" way in ATRC evaluation code.
        '''
        if database == 'PASCALContext':
            n_classes = 20
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True
             
        elif database == 'NYUD':
            n_classes = 13
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False

        else:
            raise NotImplementedError
        
        self.n_classes = n_classes + int(has_bg)
        self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = (gt != self.ignore_idx)
    
        for i_part in range(0, self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes
            
    def get_score(self, verbose=True):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        # eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)


        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            class_IoU = jac #eval_result['jaccards_all_categs']
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result


import numpy as np
import os
import json
import sys
import pprint
import random
import shutil
from PIL import Image
import glob
import copy
import torch
import cv2

# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# face-parsing.PyTorch also includes 19 attributes，but with different permutation
FFHQ_label_list = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye',
                                    'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 
                                    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 
                                    'cloth', 'hair', 'hat']  # skin-1 l_brow-2 ... 

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']
DEFAULT_SWAP_REGION_INDICES = [1, 2, 3, 5, 6, 9]


def _normalize_selected_indices(selected_indices):
    if selected_indices is None:
        return list(DEFAULT_SWAP_REGION_INDICES)
    normalized = []
    for index in selected_indices:
        index = int(index)
        if index <= 0 or index >= len(faceParser_label_list_detailed):
            continue
        if index not in normalized:
            normalized.append(index)
    return normalized


def swap_head_mask_revisit_considerGlass(source, target, hair_first=True, selected_indices=None):
    res = np.zeros_like(target)

    selected_indices = _normalize_selected_indices(selected_indices)

    target_regions = [np.equal(target, i) for i in range(12)]
    source_regions = [np.equal(source, i) for i in range(12)]

    # the target background is always preserved
    res[target_regions[0]] = 99  # a place-holder magic number 

    for class_idx in range(1, len(faceParser_label_list_detailed)):
        if class_idx in selected_indices:
            continue
        if class_idx == 4 and not hair_first:
            continue
        res[target_regions[class_idx]] = class_idx

    for class_idx in selected_indices:
        res[np.logical_and(source_regions[class_idx], np.not_equal(res, 99))] = class_idx

    if not hair_first and 4 not in selected_indices:
        res[target_regions[4]] = 4

    # the missing pixels, fill in skin
    if np.sum(res==0) != 0:
        hole_map = 255*(res==0)
        res[res==0] = 6
    else:
        hole_map = np.zeros_like(res)
        
    # restore the background
    res[res==99] = 0
     
    return res, hole_map

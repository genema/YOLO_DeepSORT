# -*- coding: utf-8 -*-
# @Author: ghma
# @Date:   2018-01-30 14:38:38
# @Last Modified by:   ghma
# @Last Modified: 2018-01-30 15:33:45
from ctypes import *
import numpy as np
import os

##############################################################
lib = CDLL("/home/ghma/deep_sort/libdarknet_for_eval.so", RTLD_GLOBAL)
##############################################################

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib.network_width.argtypes  = [c_void_p]
lib.network_width.restype   = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype  = c_int
set_gpu                     = lib.cuda_set_device
set_gpu.argtypes            = [c_int]
make_boxes                  = lib.make_boxes
make_boxes.argtypes         = [c_void_p]
make_boxes.restype          = POINTER(BOX)
free_ptrs                   = lib.free_ptrs
free_ptrs.argtypes          = [POINTER(c_void_p), c_int]
num_boxes                   = lib.num_boxes
num_boxes.argtypes          = [c_void_p]
num_boxes.restype           = c_int
make_probs                  = lib.make_probs
make_probs.argtypes         = [c_void_p]
make_probs.restype          = POINTER(POINTER(c_float))
detect                      = lib.network_predict
detect.argtypes             = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]
load_net                    = lib.load_network
load_net.argtypes           = [c_char_p, c_char_p, c_int]
load_net.restype            = c_void_p
free_image                  = lib.free_image
free_image.argtypes         = [IMAGE]
load_meta                   = lib.get_metadata
lib.get_metadata.argtypes   = [c_char_p]
lib.get_metadata.restype    = METADATA
load_image                  = lib.load_image_color
load_image.argtypes         = [c_char_p, c_int, c_int]
load_image.restype          = IMAGE
network_detect              = lib.network_detect
network_detect.argtypes     = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


def yolov2(net, meta, image, cls_num, thresh=.3, hier_thresh=.5, nms=.45, target=None):
    #print image
    if target == None:
        target = range(meta.classes)
    im    = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num   = num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    s = []
    for j in range(num):
        flag    = 0
        temp    = [0 for xx in range(6)]
        temp[2] = boxes[j].x - 0.5*boxes[j].w
        temp[3] = boxes[j].y - 0.5*boxes[j].h
        temp[4] = boxes[j].w
        temp[5] = boxes[j].h

        for i in target:
            if probs[j][i] > thresh:
                temp[1] = probs[j][i]
                temp[0] = i
                flag = 1
        if flag:
            s.append(temp)

    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)

    # return
    # -------
    # [[cls_id conf left top width height]...]
    return np.array(s)

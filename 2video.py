# -*- coding: utf-8 -*-
# @Author: ghma
# @Date:   2018-01-31 11:18:13
# @Last Modified by:   ghma
# @Last Modified time: 2018-01-31 11:26:23
import cv2
import os

videoWrite = cv2.VideoWriter("0130.avi", cv2.cv.CV_FOURCC('M','P','4','2'), 10, (1920,1088),True)

from skvideo.io import VideoWriter
#imList = os.listdir('./frame')
for i in xrange(3040):
  image = cv2.imread("./frame/{}.jpg".format(i))
  print i
  #wr.write(im)
  videoWrite.write(image)


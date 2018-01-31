# -*- coding: utf-8 -*-
# @Author:
# @Date:   2018-01-30 14:38:38
# @Last Modified by:   ghma
# @Last Modified: 2018-01-30 15:33:45
import numpy as np
import colorsys
import time
import cv2
from .image_viewer import ImageViewer


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def draw_t(detections):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape    = seq_info["image_size"][::-1]
        aspect_ratio   = float(image_shape[1]) / image_shape[0]
        image_shape    = 1024, int(aspect_ratio * 1024)
        self.viewer    = ImageViewer(update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx  = seq_info["max_frame_idx"]
        self.cent_p    = {}
        self.viewer.thickness = 2

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image
        #output_save_dir = './lab_output/'
        #pref = time.strftime("%Y%m%d%H%M%S", time.localtime)
        #cv2.imwrite('{}.jpg/{}'.format(output_save_dir+pref, time), image)

    def save_image(self, name):
        cv2.imwrite(name, self.viewer.image.copy())

    def draw_locus(self, t_id):
        for i in range(len(self.cent_p[t_id])-1):
            self.viewer.line(self.cent_p[t_id][i], self.cent_p[t_id][i+1])

    def draw_groundtruth(self, track_ids, boxes):
        '''
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))
        '''
        pass

    def draw_detections(self, detections):
        '''
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)
        '''
        pass
    '''
    def draw_t(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.point(*detection.tlwh)
    '''
    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(*track.to_tlwh().astype(np.int), label=str(track.track_id))
            #self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2], label="%d" % track.track_id)
            if track.track_id not in self.cent_p.keys():
                self.cent_p[track.track_id] = []
            else:
                x, y, w, h = track.to_tlwh().astype(np.int)
                xx = x + .5*w
                yy = y + .5*h
                flag = False
                if len(self.cent_p[track.track_id]) < 2:
                    flag = True
                else:
                    xx_prev, yy_prev = self.cent_p[track.track_id][-1]
                    prev = np.array([xx_prev, yy_prev])
                    cur  = np.array([xx, yy])
                    if np.sum(np.abs(prev-cur)) > 10:
                        flag = True
                if flag:
                    self.cent_p[track.track_id].append([xx.astype(np.int), yy.astype(np.int)])
                self.draw_locus(track.track_id)

#

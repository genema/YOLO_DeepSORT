# -*- coding: utf-8 -*-
# @Author:
# @Date:   2018-01-29 14:12:46
# @Last Modified by:   ghma
# @Last Modified time: 2018-01-31 14:50:34
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from yolov2 import yolov2 as detect_
from yolov2 import load_meta, load_net
from generate_detections import create_box_encoder, generate_detections


meta_path   = b'cfg/topsky_1016.data'
cfg_path    = b'cfg/1016.cfg'
weight_path = b'backup/1016_final.weights'


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run_ghma(sequence_dir, detection_file, output_file,
            min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance,
            nn_budget, display):

    seq_info     = gather_sequence_info(sequence_dir, detection_file)
    image_shape  = seq_info["image_size"][::-1]
    aspect_ratio = float(image_shape[1]) / image_shape[0]
    image_shape  = 1024, int(aspect_ratio * 1024)

    first_idx    = seq_info["min_frame_idx"]
    last_idx     = seq_info["max_frame_idx"]
    if os.path.isfile('0130/01/det/det.txt'):
        if not os.path.getsize('0130/01/det/det.txt'):
            detFlag = False
        else:
            detFlag = True
    else:
        detFlag = False

    if not detFlag:
        net  = load_net(cfg_path, weight_path, 0)
        meta = load_meta(meta_path)
        det_file = open('0130/01/det/det.txt', 'w')
        for idx in range(first_idx, last_idx+1):
            print(idx)
            result = detect_(net, meta, seq_info["image_filenames"][idx].encode('utf-8'), 11, target=range(8))
            for j in range(len(result)):
                det_file.write("%d,%d,%f,%f,%f,%f,%f,-1,-1,-1\n" %
                            (idx, result[j][0], result[j][2], result[j][3], result[j][4], result[j][5], result[j][1]))
        det_file.close()
    else:
        print(">> Detections already exsits, skip yolo detection step")

    #seq_info = gather_sequence_info(sequence_dir, "./temp")\
    if os.path.isfile('./temp/01.npy'):
        if not os.path.getsize('./temp/01.npy'):
            extFlag = False
        else:
            extFlag = True
    else:
        extFlag = False

    if not extFlag:
        f = create_box_encoder("resources/networks/mars-small128.ckpt-68577", batch_size=32, loss_mode="cosine")
        generate_detections(f, "./0130/", "./temp/", None)
    else:
        print(">> Features already exists, skit extraction step")
    seq_info = gather_sequence_info(sequence_dir, "./temp/01.npy")
    metric   = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker  = Tracker(metric)
    results  = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes      = np.array([d.tlwh for d in detections])
        scores     = np.array([d.confidence for d in detections])
        indices    = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            vis.save_image("./frame/{}.jpg".format(frame_idx))

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=100)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    #f = open(output_file, 'w')
    #for row in results:
    #    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #        row[0], row[1], row[2], row[3], row[4], row[5]),file=f)



def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """
    while 1:
        Run multi-target tracker on a particular sequence.

        Parameters
        ----------
        sequence_dir : str
            Path to the MOTChallenge sequence directory.
        detection_file : str
            Path to the detections file.
        output_file : str
            Path to the tracking output file. This file will contain the tracking
            results on completion.
        min_confidence : float
            Detection confidence threshold. Disregard all detections that have
            a confidence lower than this value.
        nms_max_overlap: float
            Maximum detection overlap (non-maxima suppression threshold).
        min_detection_height : int
            Detection height threshold. Disregard all detections that have
            a height lower than this value.
        max_cosine_distance : float
            Gating threshold for cosine distance metric (object appearance).
        nn_budget : Optional[int]
            Maximum size of the appearance descriptor gallery. If None, no budget
            is enforced.
        display : bool
            If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric   = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker  = Tracker(metric)
    results  = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            vis.draw_t(detections)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=10)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    track_temp = []
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
        track_temp.append([int(x) for x in row[2:6]])
    np.save('track.temp', np.array(track_temp))


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default='./0130/01')
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=False, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    '''
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
    '''
    run_ghma(args.sequence_dir, args.detection_file, args.output_file,
            args.min_confidence, args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, args.display)

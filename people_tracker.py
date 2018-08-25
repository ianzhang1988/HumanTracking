#!/usr/bin/env python
#coding=utf-8
'''
Multitarget planar tracking
==================

Example of using features2d framework for interactive video homography matching.
ORB features and FLANN matcher are used. This sample provides PlaneTracker class
and an example of its usage.

video: http://www.youtube.com/watch?v=pzVbhxx6aog

Usage
-----
plane_tracker.py [<video source>]

Keys:
   SPACE  -  pause video
   c      -  clear targets

Select a textured planar object to track by drawing a box with a mouse.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2

from imutils.object_detection import non_max_suppression
import imutils

# built-in modules
from collections import namedtuple

# local modules
#import video
#import common
#from video import presets

import json


FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

class PlaneTracker:
    def __init__(self):
        self.detector = cv2.ORB_create( nfeatures = 1000 )
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []
        self.frame_points = []

    def add_target(self, image, rect, data=None):
        '''Add a new tracking target.'''
        x0, y0, x1, y1 = rect
        raw_points, raw_descrs = self.detect_features(image)
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        descs = np.uint8(descs)
        self.matcher.add([descs])
        target = PlanarTarget(image = image, rect=rect, keypoints = points, descrs=descs, data=data)
        self.targets.append(target)

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, frame_descrs = self.detect_features(frame)
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs


class App:
    def __init__(self, src):
        #self.cap = video.create_capture(src, presets['book'])
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.paused = False
        self.tracker = PlaneTracker()

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = self.fps // 2

        print('fps', self.fps)
        print('interval',self.interval)

        cv2.namedWindow('plane')
        #cv2.namedWindow('plane2')
        #self.rect_sel = common.RectSelector('plane', self.on_rect)

        self.all_frame_positions = []

    def make_one_position(self, x, y, w, h, frame_count):
        pos = {
            u'x': int(x),
            u'y': int(y),
            u'w': int(w),
            u'h': int(h),
            u'frame_count': int(frame_count),
        }

        return pos

    def add_position_per_frame(self, pick, frame_count):
        frame_positions = []
        for (xA, yA, xB, yB) in pick:
            frame_positions.append( self.make_one_position( xA, yA, xB-xA, yB-yA, frame_count ) )
        return frame_positions

    def on_rect(self, rect):
        self.tracker.add_target(self.frame, rect)

    def run(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        frame_count = 0

        while True:
            playing = not self.paused # and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                # print('frame count',self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if not ret:
                    break
                frame_count = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                if frame_count % self.interval != 0:
                    continue
                self.frame = frame.copy()

            image = self.frame.copy()
            if playing:
                # tracked = self.tracker.track(self.frame)
                # for tr in tracked:
                #     cv2.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                #     for (x, y) in np.int32(tr.p1):
                #         cv2.circle(vis, (x, y), 2, (255, 255, 255))

                image = imutils.resize(image, width=min(1280, image.shape[1]))
                orig = image.copy()

                # detect people in the image
                # The size of the sliding window is fixed at 64 x 128 pixels
                (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                        padding=(8, 8), scale=1.05, hitThreshold=0.5)
                # print(weights)

                # draw the original bounding boxes
                for (x, y, w, h) in rects:
                    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # apply non-maxima suppression to the bounding boxes using a
                # fairly large overlap threshold to try to maintain overlapping
                # boxes that are still people
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

                # draw the final bounding boxes
                for (xA, yA, xB, yB) in pick:
                    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    cv2.circle(image,((xA+xB)//2, (yA+yB)//2),7,(255, 0, 0),3)

                self.all_frame_positions.append(self.add_position_per_frame(pick,frame_count))

            #self.rect_sel.draw(image)
            cv2.imshow('plane', image)
            # cv2.imshow('plane2', orig)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break

        self.save_position_file(self.all_frame_positions)

    def save_position_file(self, all_frame_positions):
        with open('positions.json','w') as f:
            json.dump(all_frame_positions, f)

if __name__ == '__main__':
    print(__doc__)

    import sys
    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0

    video_src = 'PNNL_Parking_LOT.avi'
    App(video_src).run()

#coding = utf-8

import cv2 as cv
import numpy as np
import json
import imutils
import time

def calc_norm_hist( img ):
    bgr_hist=np.zeros(256*3).reshape(256,3)
    for i in range(3):
        hist = cv.calcHist([img], [i], None, [256], [0,256])
        normalize_hist = np.zeros( hist.shape )
        normalize_hist = cv.normalize(hist,  normalize_hist, 0, 1, norm_type = cv.NORM_MINMAX)
        bgr_hist[:,i] = normalize_hist.ravel()
    return bgr_hist

def hist2vec(hist):
    return hist.ravel(order='F').tolist()

def show_hist( bgr_hist ):
    h = np.zeros((256,256,3))
    color=((255,0,0),(0,255,0),(0,0,255))
    for i in range(3):
        bins = np.arange(256).reshape(256,1)
        show_hist = np.int32(np.around(255 - bgr_hist[:,i].ravel()*255))
        pts = np.column_stack((bins, show_hist))
        #print(pts)
        cv.polylines(h, [pts], False, color[i])
    return h

def test_hist():
    img_name = 'tomb_raider_test.jpg'
    img = cv.imread(img_name)
    cv.namedWindow('img')
    cv.imshow('img', img)
    ch = cv.waitKey()

    bgr_hist = calc_norm_hist(img)
    h = show_hist(bgr_hist)

    print(bgr_hist[:,0].ravel()[:10])
    print(bgr_hist.ravel(order='F')[:10])
    print(hist2vec(bgr_hist)[:10])
    print(type(hist2vec(bgr_hist)[0]))

    cv.imshow('colorhist',h)
    cv.waitKey(0)

def read_positions(file = 'positions.json'):
    positions=None
    with open(file,'r') as f:
        positions = json.load(f)
    return positions

# 去除过大的和过小的目标 scale_factor_upper 最大超过平均值的倍数
def remove_unfit(positions, scale_factor_upper = 1.5, scale_factor_lower = 0.5):
    # 计算长宽平均值
    w_sum=0
    h_sum=0
    counter=0
    for frame in positions:
        for detection in frame:
            w = detection['w']
            h = detection['h']
            w_sum+=w
            h_sum+=h
            counter+=1

    w_average = w_sum // counter
    h_average = h_sum // counter

    print( 'avg width: %s, avg height: %s'% (w_average, h_average ))

    w_upper_bound = w_average*scale_factor_upper
    w_lower_bound = w_average*scale_factor_lower
    h_upper_bound = h_average*scale_factor_upper
    h_lower_bound = h_average*scale_factor_lower

    fit_positions = []
    for frame in positions:
        fit_detection = []
        for detection in frame:
            w = detection['w']
            h = detection['h']
            if (w > w_upper_bound
                or w < w_lower_bound
                or h > h_upper_bound
                or h < h_lower_bound):
                print('delete', detection)
                pass
            else:
                fit_detection.append(detection)

        if len(fit_detection) > 0:
            fit_positions.append(fit_detection)

    # 检查异常 稳定后删除
    for frame in fit_positions:
        for detection in frame:
            w = detection['w']
            h = detection['h']
            if (w > w_upper_bound
                or w < w_lower_bound
                or h > h_upper_bound
                or h < h_lower_bound):
                print('unexpected', detection)

        if len(frame) < 1:
            print('unexpected frame')

    return fit_positions

def re_calc_pos(x,y,w,h, retain_length = 7/8):
    _x = _y = _w = _h = 0

    _w = int(w*retain_length)
    _h = int(h*retain_length)

    _x = x + (w - _w)//2
    _y = y + (h - _h)//2

    return _x,_y,_w,_h


positions = read_positions()
fit_positions = remove_unfit(positions)

cap = cv.VideoCapture( 'PNNL_Parking_LOT.avi' )

pos_index = 0
while pos_index < len(fit_positions):
    seg = fit_positions[pos_index]
    if len(seg)< 1:
        continue

    needed_frame_count =seg[0]['frame_count']

    ret, frame = cap.read()
    if not ret:
        break
    frame_count = cap.get(cv.CAP_PROP_POS_FRAMES)
    if frame_count < needed_frame_count:
        continue

    frame_tmp = frame.copy()
    frame_tmp = imutils.resize(frame_tmp, width=min(1280, frame_tmp.shape[1]))
    pos_index += 1

    for detection in seg:
        x = detection['x']
        y = detection['y']
        w = detection['w']
        h = detection['h']

        cv.rectangle(frame_tmp, (x, y), (x + w, y + h), (0, 0, 255), 2)

        x, y, w, h = re_calc_pos(x,y,w,h,1/2)

        crop_img = frame_tmp[y:y+h,x:x+w]

        # cv.imshow('test_crop', imutils.resize(crop_img, width=200))
        # cv.waitKey(333)

        bgr_hist = calc_norm_hist(crop_img)
        hist_feature = hist2vec(bgr_hist)
        detection['hist_feature'] = hist_feature

    # cv.imshow('draw_rect', frame_tmp)
    #cv.waitKey(0)

with open('pos_with_hist.json','w') as f:
    json.dump(fit_positions,f)

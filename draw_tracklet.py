# coding=utf-8

import json
import cv2
import numpy as np

def read_data():
    file_name = 'tracklet.json'
    data = None
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

data = read_data()

img = np.zeros((1920,1080,3),dtype=np.uint8)

color_index = -1

def get_color():
    color_list = [
        (192, 192, 192),
        (255, 153, 18),
        (255, 97, 0),
        (65, 105, 255),
        (252, 230, 201),
        (0, 255, 0),
        (245, 245, 245),
        (255, 255, 0),
        (64, 224, 208),
        (255, 0, 0),
    ]

    global color_index
    color_index+=1
    if color_index >= len(color_list):
        color_index = 0

    return color_list[color_index]


for cluster in data:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    for person in cluster[:3]:
        pts = []
        for data in person:
            x = data['x']
            y = data['y']
            w = data['w']
            h = data['h']
            pts.append((x+w/2, y+h/2))

        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img,[pts],False,get_color(),3)

    cv2.imshow('img', img)
    cv2.waitKey(0)






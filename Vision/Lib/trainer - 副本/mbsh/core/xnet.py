# -*- coding: utf-8 -*-
"""

"""

from keras.layers import *
from keras.models import *

import cv2
import numpy as np

from mbsh.core.images import read_img_file, cv2FindContours


def mini_unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(pool3))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


# 根据轮廓点列表计算轮廓面积
def area_cal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])
    return area


def area_sort_key(elem):
    return area_cal([elem])


# (0, 0, 255) #红色
def draw_outline_rect(img, ctrs, color=(0, 0, 255), line_thickness=2, horizontal=True, draw_line=True):
    rectangle_list = []

    if horizontal:
        # 画水平方框
        for c in ctrs:
            # find bounding box coordinates
            x, y, w, h = cv2.boundingRect(c)
            if draw_line:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, line_thickness)
            rectangle_list.append([x,y,w,h])
    else:
        for c in ctrs:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 画外框
            if draw_line:
                cv2.drawContours(img, [box], 0, color, line_thickness)

    return rectangle_list


def get_points(x1, y1, x2, y2, n):
    points_list = []
    for i in range(1, n):
        x = x1 + int(i * (x2 - x1) / n)
        y = y1 + int(i * (y2 - y1) / n)
        points_list.append([x, y])

    return points_list


# gap控制点的稀疏程度，越大越稀疏
def draw_points_between_2points(img, st_x, st_y, end_x, end_y, point_size, color, gap):
    tmp = (end_x - st_x) ** 2 + (end_y - st_y) ** 2
    tmp = tmp ** 0.5

    count = int(tmp // gap)

    if count < 1:
        return [st_x, st_y]
    elif count == 1:
        points_list = [[end_x, end_y]]
    else:
        points_list = get_points(st_x, st_y, end_x, end_y, count)

    for k in range(0, len(points_list)):
        cv2.circle(img, tuple(points_list[k]), point_size, color, -1)

    return points_list[-1]


# gap控制点的稀疏程度，越大越稀疏
# (255, 0, 0) #蓝色
def draw_outline_points(img, ctrs, color=(255, 0, 0), point_size=2, gap=7):
    for i in range(len(ctrs)):
        # 起点
        draw_point = ctrs[i][0][0].tolist()
        cv2.circle(img, tuple(draw_point), point_size, color, -1)

        # 2点间画点
        for j in range(1, len(ctrs[i])):
            # 计算距离
            x = ctrs[i][j][0][0]
            y = ctrs[i][j][0][1]
            draw_point = draw_points_between_2points(img, draw_point[0], draw_point[1], x, y, point_size, color, gap)

        # 首尾，起点：ctrs[i][0][0]，终点：draw_point
        x = ctrs[i][0][0][0]
        y = ctrs[i][0][0][1]
        draw_points_between_2points(img, draw_point[0], draw_point[1], x, y, point_size, color, gap)


def xnet_predict_img(model, src_img, model_input_img_size, predict_area_num=1, gray_threshold=127, verbose=1, min_area=None):
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]

    img = src_img
    img = img[:, :, :: -1]

    img = cv2.resize(img, model_input_img_size)
    img = img.astype(np.float64)
    img = img / 255.

    results = model.predict(np.array([img]), verbose=verbose)
    gray_img = results[0][:, :, 0]
    bw_heatmap = np.uint8(255 * gray_img)
    bw_heatmap[bw_heatmap <= gray_threshold] = 0
    bw_heatmap[bw_heatmap > gray_threshold] = 255
    ai_ctrs = cv2FindContours(bw_heatmap.copy())

    ai_ctrs.sort(key=area_sort_key, reverse=True)
    ai_ctrs = ai_ctrs[:predict_area_num]
    if min_area:
        temp_ctrs = []
        for ctrs in ai_ctrs:
            if area_cal([ctrs]) >= min_area:
                temp_ctrs.append(ctrs)
        ai_ctrs = temp_ctrs
    x_rate = src_img_w / model_input_img_size[0]
    y_rate = src_img_h / model_input_img_size[0]

    min_area_num = min(len(ai_ctrs), predict_area_num)
    for i in range(min_area_num):
        ai_ctrs[i][:, :, 0] = ai_ctrs[i][:, :, 0] * x_rate
        ai_ctrs[i][:, :, 1] = ai_ctrs[i][:, :, 1] * y_rate

    return ai_ctrs


def xnet_predict_file(model, file, model_input_img_size, predict_area_num=1, gray_threshold=127, verbose=1, min_area=None):
    src_img = read_img_file(file)
    ai_ctrs = xnet_predict_img(model, src_img, model_input_img_size, predict_area_num, gray_threshold, verbose, min_area)
    return src_img, ai_ctrs


def sparse_get_point(st_x, st_y, end_x, end_y, gap=7):
    distance = (end_x - st_x) ** 2 + (end_y - st_y) ** 2
    distance = distance ** 0.5

    count = int(distance // gap)

    if count < 1:
        return None, [st_x, st_y]
    elif count == 1:
        ps = [[end_x, end_y]]
    else:
        ps = [[st_x, st_y], [end_x, end_y]]

    return ps, ps[-1]


# 轮廓点稀疏化
def sparse_ctrs(in_ctrs, gap=7):
    out_ctrs = []

    for ctrs in in_ctrs:
        sub_out_ctrs = []
        # first point
        start_point = ctrs[0][0].tolist()
        sub_out_ctrs.append(start_point)
        for i in range(1, len(ctrs)):
            end_x = ctrs[i][0][0]
            end_y = ctrs[i][0][1]
            points, start_point = sparse_get_point(start_point[0], start_point[1], end_x, end_y, gap)
            if points:
                sub_out_ctrs.extend(np.array(points))

        out_ctrs.append(np.array(sub_out_ctrs))

    return out_ctrs


# -*- coding: utf-8 -*-
# @File   : images.py
# @Author : zhkuo
# @Time   : 2020/11/20 16:44
# @Desc   : 图像相关处理

import os
import cv2
from PIL import Image
import numpy as np


def cv2_imread(path, mode=cv2.IMREAD_COLOR):
    """
    cv2 模式读取图片
    """
    img_read = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    return img_read


def cv2_imwrite(path, img_write):
    """
    cv2 模式保存图片
    """
    suffix = os.path.splitext(path)[-1]
    cv2.imencode(suffix, img_write)[1].tofile(path)


def resize_image(img, size):
    """
    按比例 resize 图片，非正方形补黑边
    :param img: 输入为 cv2 模式
    :param size:
    :return: 输出为 cv2 模式
    """
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    img = np.array(new_image)
    return img[:, :, ::-1]


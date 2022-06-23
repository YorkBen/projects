# coding=utf-8
from multiprocessing import Process
import re
import shutil
import os
from collections import Counter
import numpy as np

from mbsh.core.images import join_processes, cal_file_name, read_img_file, resize_image
from mbsh import logger, g_temp_dir
import cv2

__author__ = 'study'


def crop_dir(from_dir, target_path, n_process=1, size=(360, 360)):
    """
    处理从内镜中心导出的图片
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    all_files = []
    for root, dirs, files in os.walk(from_dir, followlinks=True):
        for name in files:
            path_join = os.path.join(root, name)
            all_files.append(path_join)
    total_files = len(all_files)

    logger.info("start to prepare images ,total files %s " % total_files)
    length = total_files / float(n_process)

    indices = [int(round(i * length)) for i in range(n_process + 1)]
    sub_list = [all_files[indices[i]:indices[i + 1]] for i in range(n_process)]
    processes = [Process(target=crop_file_list, args=[sub_list[i], target_path, size]) for i
                 in
                 range(n_process)]
    join_processes(processes)


def crop_file_list(files, target_dir, size=(360, 360), remain_filename=False):
    for imgpath in files:
        file = os.path.split(imgpath)[-1]
        if os.path.isfile(imgpath):
            if remain_filename:
                target_name = file.replace('TIF', 'jpg').replace('bmp', 'jpg')
            else:
                target_name = cal_file_name(file).replace('TIF', 'jpg').replace('bmp', 'jpg')
            target_file = os.path.join(target_dir, target_name)

            try:
                shutil.copy(imgpath, target_file)
                result, img = crop_target(target_file)
                if size:
                    # img = cv2.resize(img, size)
                    # 保持宽高比
                    img = resize_image(img, size)

                # 改用imencode方式，以支持中文路径名、文件名
                # cv2.imwrite(target_file, img)
                cv2.imencode('.jpg', img)[1].tofile(target_file)
            except:
                logger.info("crop file fail:%s" % imgpath)
                if os.path.exists(target_file):
                    os.remove(target_file)
        else:
            logger.error("is not a file %s " % imgpath)


def crop_res(src_dir, target_dir):
    """
    处理已经分类的到处图片
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir)
    _dirs = os.listdir(src_dir)

    dirs = sorted(_dirs, key=lambda x: (int(re.sub('\D', '', x)), x))

    logger.debug(dirs)
    logger.debug([re.sub('\d+\W', '', x) for x in dirs])

    processes = []
    for i, dir_name in enumerate(dirs):
        from_dir = src_dir + "/" + dir_name
        fold = dir_name if dir_name.isdigit() else str(i)
        target_path = target_dir + "/" + fold
        logger.info('copy res from %s to %s ' % (from_dir, target_path))
        p1 = Process(target=crop_dir, args=(from_dir, target_path))
        processes.append(p1)
    join_processes(processes)
    logger.info('copy res finished')


def crop_target(img_path):
    """
    剪裁内镜影像
    @:return bool,img

    """
    img = read_img_file(img_path)
    if img is None:
        logger.warn('error img %s' % img_path)
        return False, []
    return crop_img(img, img_path)


def detect_area(img, save_img=False):
    """
    检测内镜影像边框
    @:return bool,img

    """
    if img is None:
        return False, []
    img_path = os.path.join(g_temp_dir, 'input_src.jpg')
    # 保存原图
    if save_img:
        cv2.imwrite(img_path, img)

    src_img_W = img.shape[0]
    src_img_H = img.shape[1]
    img_src = img.copy()
    gray_black = np.ones((60, src_img_H), dtype=np.uint8)

    gray_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到灰度图颜色数量最多的颜色，作为图片底部添加的边，以找到正确的裁剪边界
    color_count_dict = Counter(gray_src.reshape(-1))
    color_count_list = list(color_count_dict.items())
    color_count_list.sort(key=lambda x: x[1], reverse=True)

    # 在原图片底部加固定高度黑边,生成新图片便于找到裁切的边界
    gray_merge = np.concatenate((gray_src, gray_black * int(color_count_list[0][0])))

    try:
        gray = gray_merge
        bw = cv2.adaptiveThreshold(gray, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # cv2.__version__为4、2的返回2个值，为3的返回3个值
        try:
            _, ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        if len(rects) == 0:
            return False, []
        rects.sort(key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = rects[0]
        if y + h > src_img_H:
            h = src_img_H - y
        roi = img_src[y:y + h, x:x + w]
        left = 0
        # 有效面积占比
        rate = w * h * 1.0 / (src_img_W * src_img_H)

        if 0.20 < rate:
            img1 = roi
            # 尝试切除左边的黑边
            if img1.shape[1] > src_img_W * 0.9:
                left = find_left_black(img1)
                if left > src_img_W * 0.4:
                    logger.warn('too much black ')
                    left = 0
            rec = [x + left, y, x + w, y + h]

            # 保存裁切后的图片
            if save_img:
                img_path = os.path.join(g_temp_dir, 'input_area.jpg')
                cv2.imwrite(img_path, img_src[rec[1]:rec[3], rec[0]:rec[2]])
            return True, rec
        else:
            logger.debug("detect_area roi fail,w:{},h:{},rate:{}".format(w, h, rate))
    except:
        logger.exception("detect_area img fail:")

    return False, []


def crop_img(img, img_path):
    """
    剪裁内镜影像
    @:return bool,img

    """
    return True, img
    if img is None:
        return False, []

    # src_img_W是高 src_img_H是宽
    src_img_W = img.shape[0]
    src_img_H = img.shape[1]

    img_src = img.copy()
    gray_black = np.ones((60, src_img_H), dtype=np.uint8)

    gray_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到灰度图颜色数量最多的颜色，作为图片底部添加的边，以找到正确的裁剪边界
    color_count_dict = Counter(gray_src.reshape(-1))
    color_count_list = list(color_count_dict.items())
    color_count_list.sort(key=lambda x: x[1], reverse=True)

    # 在原图片底部加固定高度黑边,生成新图片便于找到裁切的边界
    gray_merge = np.concatenate((gray_src, gray_black * int(color_count_list[0][0])))
    try:
        gray = gray_merge
        bw = cv2.adaptiveThreshold(gray, 50, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        img2, ctrs, hier = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        if len(rects) == 0:
            return False, []
        rects.sort(key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = rects[0]
        if y + h > src_img_H:
            h = src_img_H - y
        roi = img_src[y:y + h, x:x + w]
        left = 0
        # 有效面积占比
        rate = w * h * 1.0 / (src_img_W * src_img_H)
        _img = img_src
        if 0.40 < rate:
            img1 = roi
            # 尝试切除左边的黑边
            if img1.shape[1] > src_img_W * 0.9:
                left = find_left_black(img1)
                if left > src_img_W * 0.4:
                    logger.warn('too much black %s' % (img_path))
                    left = 0
            rec = [x + left, y, x + w, y + h]
            _img = img_src[rec[1]:rec[3], rec[0]:rec[2]]
        return True, _img
    except:
        logger.exception("crop img fail: %s" % (img_path))
    return False, []


def find_left_black(img):
    """
    查找左边的黑边,胃镜左边有黑边
    """
    gray_image = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols, _ = img.shape
    _img = gray_image[:, :]
    half = cols * 2 // 3
    globe_mean = np.mean(gray_image)
    for i in range(half, cols - 1):
        _shape_ = _img[:, cols - (i + 1):cols - i]
        mean = np.mean(_shape_)

        if mean < 18:
            # _max = np.max(_shape_)
            # std = np.std(_shape_)
            # print("small mean,left=%s ,mean=%s ,std=%s,globe_mean=%s,max=%s" % (
            #     cols - i, mean, std, globe_mean, _max))
            return cols - i

    for i in range(half, cols - 1):
        _shape_ = _img[:, cols - (i + 1):cols - i]
        mean = np.mean(_shape_)
        if mean < 0.6 * globe_mean:
            # _max = np.max(_shape_)
            # std = np.std(_shape_)
            # print("small globe_mean,left=%s ,mean=%s ,std=%s,globe_mean=%s,max=%s" % (
            #     cols - i, mean, std, globe_mean, _max))
            return cols - i

    return 0

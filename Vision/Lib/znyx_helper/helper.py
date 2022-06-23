# encoding: utf-8

"""
调试 jupyter 辅助函数
tf1.12.0
"""
import sys

import cv2
import os
import numpy as np
from model_encrypt import decrypt_file, load_encrypt_model
from PIL import Image
import pickle
from tensorflow.python.keras.models import model_from_json
# sys.path.insert(0, r'E:\projects\znyx-trainer\books\zhkuo_tmp\znyx-model\trunk\znyx-model')
from mbsh.core.yolo import YOLO




def cv2_imread(path, mode=cv2.IMREAD_COLOR):
    """
    读取路径下图片
    """
    img_read = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    return img_read


def cv2_imwrite(path, img_write):
    """
    存储图片到指定路径
    """
    suffix = os.path.splitext(path)[-1]
    cv2.imencode(suffix, img_write)[1].tofile(path)


def resize_image(img, size):
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


def resize_img(image, size):
    """
    resize 到短边为size大小
    """
    (h, w) = image.shape[:2]

    if w < h:
        r = size / float(w)
        new_h = int(h * r)
        new_w = size
    else:
        r = size / float(h)
        new_h = size
        new_w = int(w * r)

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img


def center_crop(image, size=224):
    h, w, _ = image.shape
    if w > h:
        index = int((w - h) / 2)
        img = image[:, index: index + size, :]
    else:
        index = int((h - w) / 2)
        img = image[index: index + size, :, :]
    return img


def load_model(model_path):
    """
    加载加密后的模型
    """
    model = load_encrypt_model(model_path)
    return model


def load_cnn_mode(json_path, weight_path):
	"""
	加载未加密的cnn模型
	"""
	_file = open(json_path, 'r')
	model = model_from_json(_file.read())
	model.load_weights(weight_path)
	return model


def load_lstm_model(json_path, weight_path):
    """
    加载 LSTM 模型
    """
    with open(json_path, 'r') as _file:
        lstm_model = model_from_json(_file.read())
        lstm_model.load_weights(weight_path)
    return lstm_model


def load_yolo3_model(file, image_size=(352, 352)):

    if not os.path.exists(file):
        print(" can not find file %s " % os.path.abspath(file))
        return
    print("  find file %s " % os.path.abspath(file))

    model = YOLO(model_path=file, model_image_size=image_size)
    return model


def is_valid_box(img, box):
    if len(img) == 0 or len(box) == 0:
        return False
    target = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
    if len(target) == 0 or len(target[0]) == 0:
        return False
    avg = float(np.max(target)) - float(np.min(target))

    valid = avg > 50
    # if  valid:
    #     out_path = '%s/box/c_%s_%s' % (g_temp_dir, int(avg), cal_file_name('2.jpg'))
    #     cv2.imwrite(out_path, target)

    return valid


def test_bbox(yolo_model, img, min_ratio=0.0, max_ratio=1.0, min_level=0.01):

    img = cv2.resize(img, yolo_model.model_image_size)
    size = max(img.shape[0], img.shape[1])

    boxes, scores, labels = yolo_model.predict(img)

    check_boxes = []
    max_size = max_ratio * size
    min_size = min_ratio * size
    max_score = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < min_level:
            break

        if (box[2] - box[0]) > max_size or (box[3] - box[1]) > max_size:
            continue

        if (box[2] - box[0]) < min_size or (box[3] - box[1]) < min_size:
            continue
        # ignore other label
        if label > 0:
            continue
        if not is_valid_box(img, box):
            continue
        bbox = [float(x) * 224 / size for x in box]
        check_boxes.append(bbox)
        max_score = max(score, max_score)

    predictions = [1 - max_score, max_score]
    item = max_score * 100
    index = 1 if len(check_boxes) > 0 else 0

    return index, check_boxes


def predict(model, img, threshold=0.5):
    """
    进行预测
    """
    input_img_size = (int(model.input.shape[1]), int(model.input.shape[2]))

    img = cv2.resize(img, input_img_size)
    x = np.array(img.copy())
    input_data = np.expand_dims(x, axis=0)
    predictions = model.predict(input_data)[0]

    if len(predictions) == 1:
        v = predictions[0] * 100
        confidence, class_index = (v, 1) if v > threshold else (100 - v, 0)
    else:
        confidence = int(np.max(predictions) * 100)
        class_index = int(np.argmax(predictions))

    return class_index, confidence


def test_cnn(cnn_model, img, tr=0.5):
    input_img_size = (int(cnn_model.input.shape[1]), int(cnn_model.input.shape[2]))

    _img = resize_image(img, input_img_size)

    x = np.array(_img)
    data = np.expand_dims(x, axis=0)
    predictions = cnn_model.predict(data)[0]

    if len(predictions) == 1:
        src_confidence = int(predictions[0] * 100)
        class_index = 0 if predictions[0] < tr else 1
        confidence = predictions[0] if class_index == 1 else 1 - predictions[0]
    else:
        class_index = int(np.argmax(predictions))
        confidence = np.max(predictions)
        src_confidence = confidence

    confidence = int(confidence * 100)
    confidence = min(99, confidence)

    return class_index, confidence, src_confidence, predictions


def predict_nbi_ca(model, img, threshold=0.5):
    """
    进行预测
    """
    input_img_size = (int(model.input.shape[1]), int(model.input.shape[2]))

    img = cv2.resize(img, input_img_size)
    x = np.array(img.copy())
    input_data = np.expand_dims(x, axis=0)
    predictions = model.predict(input_data)[0]
    confidence = predictions[0]

    confidence = int(confidence * 100)
    src_conf = confidence
    # 限制最大值为99
    confidence = min(99, confidence)
    class_index, confidence = (1, confidence) if confidence > threshold * 100 else (0, 100 - confidence)

    return class_index, confidence, src_conf


def predict_center_nbi_ca(model, img, threshold=0.5):
    """
    进行预测
    """
    input_img_size = (int(model.input.shape[1]), int(model.input.shape[2]))

    img = resize_img(img, 224)
    #  取中间预测一次
    img = center_crop(img, size=224)

    _img = resize_image(img, input_img_size)

    x = np.array(_img)
    data = np.expand_dims(x, axis=0)

    predictions = model.predict(data)[0]
    confidence = predictions[0]
    confidence = int(confidence * 100)
    # 限制最大值为99
    confidence = min(99, confidence)
    class_index, confidence = (1, confidence) if confidence > threshold * 100 else (0, 100 - confidence)

    return class_index, confidence


def dhash(img_src, hash_size=8):
    img = cv2.resize(img_src, (360, 360))
    img = img[30:330, 30:330]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_RGB)
    # Grayscale and shrink the image in one step.
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
    #     pixels = list(image.getdata())
    # Compare adjacent pixels.
    hash = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            if pixel_left > pixel_right:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def Hamming_distance(hash_target, hash_match):
    num = 0
    if len(hash_target) != len(hash_match) or len(hash_target) * len(hash_match) == 0:
        return -1
    for index in range(len(hash_target)):
        if hash_target[index] != hash_match[index]:
            num += 1
    return num


def save_cache(data, file_path):
    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()


def load_cache(file_path):
    data = dict()
    if os.path.isfile(file_path):
        file = open(file_path, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        print('file not exist: ' + file_path)
    return data


class TimesActivator1:
    """
    以出现次数为依据的激活器
    """

    def __init__(self, parts_levels, activate_count=3, total_count=5):
        self.last_big_part = []
        self.total_count = total_count
        self.active_labels = [None] * len(parts_levels)
        self.parts_levels = parts_levels
        self.active_parts_counts = [activate_count] * len(parts_levels)
        self.has_activate = False
        self.active_label = None

    def reset(self):
        self.last_big_part = []

    def add_label(self, img_result, is_check=False):
        if not img_result:
            return
        self.has_activate = False
        self.active_label = None
        self.last_big_part.append(img_result)
        if is_check:
            self.check_label(self.last_big_part, img_result)

    def check_label(self, img_result_list, img_result):

        img_result_list = img_result_list[-self.total_count:]
        class_level = self.parts_levels[img_result['ca_label']]
        count = 0
        for in_label in img_result_list:
            if in_label['ca_conf'] > class_level and img_result['ca_label'] == in_label['ca_label']:
                count += 1

        active_level = self.active_parts_counts[img_result['ca_label']]
        if count >= active_level:
            self.has_activate = True
            self.active_label = img_result['ca_label']






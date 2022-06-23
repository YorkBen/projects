import os
import Levenshtein
import cv2
import time
import numpy as np
import re

res_set = [
            # 奥巴
            'CF-HQ290I', 'GIF-H260Z',
            # 富士
                    '1C397G011', '1C604G147', '1C0700G106', 'EC-L590WM',
                    '1G388G162', '1G389G151', '5C362A007', 'EC-590WM', 'EG-590WR',
                    '4C736G076', '5C735G277', '2C719G956', '2C719G957', '2C719G959',
                    '1C727G230', '1C730G146', '1G403G189', '2G400G521', '2G403G013',
                    '4G407G226',
            # 开立
            'EG-500', 'EG-550']


def load_ocr_txt(type='res'):
    results = []
    # pattern = re.compile(r'[A-Z0-9-]')
    with open('%s.txt' % type) as f:
        for l in f.readlines():
            s = l.strip()
            if len(s) > 0:
                arr = s.split(' ')
                for a in arr:
                    results.append(a)
    return results


def predict(target_documents, querys):
    results = []
    for query in querys:
        max = (0.0 , '')    #（相似度， 文档）
        for doc in target_documents:
            similar = Levenshtein.ratio(query, doc)
            if similar > max[0]:
                max = (similar, doc)
        results.append((query, max[1], max[0]))

    results = sorted(results, key=lambda x: x[2], reverse=True)

    if len(results) > 0 and results[0][2] >= 0.5:
        return results[0]
    else:
        return None


def load_ocr_box():
    boxes, str = [], ''
    with open('res.box') as f:
        for l in f.readlines():
            if len(l.strip()) > 0:
                arr = l.strip().split(' ')
                boxes.append(arr[1:5])
                str = str + arr[0]
    return str, boxes


def find_txt_box(txt, str, boxes):
    txt = txt.replace(' ', '')
    start = str.find(txt)
    end = start + len(txt)
    xmins = sorted([int(box[0]) for box in boxes[start:end]])
    xmaxs = sorted([int(box[2]) for box in boxes[start:end]])
    ymins = sorted([int(box[1]) for box in boxes[start:end]])
    ymaxs = sorted([int(box[3]) for box in boxes[start:end]])
    median_l = len(ymins) // 2

    return (xmins[0], ymins[median_l], xmaxs[-1], ymaxs[median_l])


def crop_txt_img(img_path, box):
    img = cv2.imread(img_path)
    # print(img.shape)
    h, w, _ = img.shape
    # box坐标从图像下方到上方，img是从上到下，需要颠倒y
    box_reverse = (max(box[0]-5, 0), max(h-1 - box[3] - 2, 0), min(box[2]+5, w-1), min(h-1 - box[1] + 2, h-1))
    crop_img = img[box_reverse[1]:box_reverse[3]+1, box_reverse[0]:box_reverse[2]+1]
    # crop_img = preprocess_img(crop_img)
    cv2.imwrite('tmp.jpg', crop_img)


def preprocess_img(src_img):
    """
    图像预处理-腐蚀膨胀，提升识别精度
    :param src_img:
    :return:
    """
    # PIL -> cv2
    img = cv2.cvtColor(src_img,cv2.COLOR_RGB2GRAY)
    # 灰度化
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 腐蚀膨胀
    # k = np.ones((1, 2), np.uint8)
    # img = cv2.erode(img, kernel=k)
    # img = cv2.dilate(img, kernel=k)
    # img = cv2.resize(img, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


def crop_raw_img(src_img_path, img_path, crop=True):
    img = cv2.imread(src_img_path)
    img_h, img_w, _ = img.shape

    if crop:
        if '奥巴' in src_img_path:
            img = img[:, :int(img_w / 2)]
        elif '富士4450' in src_img_path:
            img = img[int(img_h * 0.4):int(img_h * 0.8), :int(img_w * 0.3)]
            # img = img[int(img_h * 0.3):int(img_h * 0.7), :int(img_w * 0.5)]
        elif '富士7000' in src_img_path:
            img = img[:, int(img_w / 2):]

    # img = preprocess_img(img)

    cv2.imwrite(img_path, img)


def process_one(in_img_path):
    img_path = '1.jpg'
    # 第0步，根据图片类型裁剪图并保存到img_path
    crop_raw_img(in_img_path, img_path)

    # 第一步，识别全图，根据识别到的文字根据最小误差找出候选文字
    os.system('tesseract %s res -l eng+fushi.exp3' % img_path)
    txts = load_ocr_txt()
    results = predict(res_set, txts)
    print(results)
    # 没有结果直接返回
    if results is None:
        crop_raw_img(in_img_path, img_path, False)
        os.system('tesseract %s res -l eng+fushi.exp3' % img_path)
        txts = load_ocr_txt()
        results = predict(res_set, txts)
        if results is None:
            return ''

    txt = results[0]
    # 如果已经相等，直接返回
    if results[2] == 1:
        return txt, 1

    # 第二步，同一命令，识别box，根据候选文字，找出其对应的box区域
    os.system('tesseract %s res -l eng+fushi.exp3 makebox' % img_path)
    str, boxes = load_ocr_box()
    box = find_txt_box(txt, str, boxes)
    # print(box)
    crop_txt_img(img_path, box)

    # 截取box区域保存为图片，重新识别
    os.system('tesseract tmp.jpg res2 -l fushi.exp3 --psm 6')
    txts = load_ocr_txt('res2')
    results = predict(res_set, txts)
    print(results)
    return results[0], results[2]


def stat_by_type(type):
    cr_ct, full_cr_ct, total_ct = 0, 0, 0
    write_arr = []
    print('process %s' % type)
    for dirpath, dirnames, filenames in os.walk(r'\\192.168.0.110\temp\ocr\%s' % type):
        for dirname in dirnames:
            tag_full = dirname.split('#')[0]
            tag = tag_full[:2]
            dir_path = os.path.join(dirpath, dirname)

            for file in os.listdir(dir_path):
                if not file.split('.')[-1].lower() in ['jpg', 'png']:
                    continue

                img_path = os.path.join(dir_path, file)
                # starttime = time.time()
                try:
                    txt, score = process_one(img_path)
                except:
                    txt = 'FAILED!'
                    print(img_path)
                # endtime = time.time()
                # print('time cost: %s' % (endtime - starttime))
                total_ct = total_ct + 1
                if score == 1 or txt == tag_full:
                    cr_ct = cr_ct + 1
                    full_cr_ct = full_cr_ct + 1
                elif txt.startswith(tag):
                    cr_ct = cr_ct + 1
                    write_arr.append('%s:%s,%s,%s' % (img_path, txt, tag, tag_full))
                else:
                    write_arr.append('%s:%s,%s,%s' % (img_path, txt, tag, tag_full))

    with open('results.txt', 'a+') as f:
        for l in write_arr:
            f.write(l + '\n')
        f.write('%s:%s,%s,%s\n' % (type, total_ct, cr_ct, full_cr_ct))


if __name__ == "__main__":
    # list = [
    #     r'\\192.168.0.110\temp\ocr\富士4450合集1\1C397G011#001_0_008_2022_03_31_11_09_247-1\103976011#0413_17_33_45_984550_ocr_img.jpg',
    # ]
    # for f in list:
    #     process_one(f)
    process_one(r'\\192.168.0.110\temp\ocr\富士4450合集1\1C397G011#001_0_005_2022_04_01_09_03_500-1\#0413_17_29_02_166780_ocr_img.jpg')


    # stat_by_type('奥巴合集')
    # stat_by_type('富士4450合集1')
    # stat_by_type('富士4450合集2')
    # stat_by_type('富士4450合集3')
    # stat_by_type('富士7000合集')






















#

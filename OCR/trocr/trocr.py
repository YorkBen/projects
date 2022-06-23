from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image, ImageOps
import time
import cv2
import numpy as np
import os

# load image from the IAM dataset
# url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
# img = "test1.png"
# image = ImageOps.invert(Image.open(img).convert("RGB"))
# url = "https://pics5.baidu.com/feed/58ee3d6d55fbb2fb9fb3ff1da0b8f9a24723dc1b.jpeg?token=3d4658f353c1912e6df7315bcec9b650"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# img_path = r'\\192.168.0.110\temp\ocr\奥巴合集\CF-H290I#01.0000000075380.0125.1536718046\CF-#0413_17_57_40_239297_ocr_img.jpg'
# img_path = r'\\192.168.0.110\temp\ocr\奥巴合集\GIF-H260Z#01.0000000130537.0093\GIF-#0413_18_01_03_490526_ocr_img.jpg'

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")


def get_bound(type, src_img):
    """
    不同的型号图片，文字位置不同
    """
    img_w, img_h = src_img.size[1], src_img.size[0]
    if type == '奥巴合集':
        h_start, h_end = int(img_h / 4), int(img_h / 3.3)
        w_start, w_end = 0, img_w // 4
    elif type == '富士4450合集1':
        h_start, h_end = int(img_h * 0.495), int(img_h * 0.507)
        w_start, w_end = img_w * 0.17, img_w * 0.23
    elif type == '富士4450合集2':
        h_start, h_end = int(img_h * 0.495), int(img_h * 0.514)
        w_start, w_end = img_w * 0.17, img_w * 0.265
    elif type == '富士4450合集3':
        h_start, h_end = int(img_h * 0.495), int(img_h * 0.507)
        w_start, w_end = img_w * 0.17, img_w * 0.23
    elif type == '富士7000合集':
        h_start, h_end = int(img_h * 0.25), int(img_h * 0.267)
        w_start, w_end = img_w * 0.728, img_w * 0.82

    bounds=(w_start,h_start,w_end,h_end)
    return bounds

def get_txt(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def get_txt_tessocr(image):
    image.save("tmp.jpg")
    # os.system('tesseract tmp.jpg res -l fushi.exp3 --psm 6')
    os.system('tesseract tmp.jpg res -l fushi.exp3 --psm 12')
    # os.system('tesseract tmp.jpg res -l eng --psm 6')
    txt = ''
    if os.path.exists('res.txt'):
        with open('res.txt') as f:
            for line in f.readlines():
                txt = line.strip()
                break

    return txt


def process_img(src_img):
    """
    图像预处理-腐蚀膨胀，提升识别精度
    :param src_img:
    :return:
    """
    # PIL -> cv2
    img = cv2.cvtColor(np.asarray(src_img),cv2.COLOR_RGB2GRAY)
    # 灰度化
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 腐蚀膨胀
    k = np.ones((1, 2), np.uint8)
    img = cv2.erode(img, kernel=k)
    img = cv2.dilate(img, kernel=k)
    img = cv2.resize(img, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(img)


def stat_ab():
    correct_ct, total_ct = 0, 0
    for dirpath, dirnames, filenames in os.walk(r'\\192.168.0.110\temp\ocr\奥巴合集'):
        for filename in filenames:
            starttime = time.time()
            tag = filename.split('-')[0]
            img_path = os.path.join(dirpath, filename)
            src_img = ImageOps.invert(Image.open(img_path).convert("RGB"))
            image = src_img.crop(get_bound("奥巴合集", src_img))
            image = process_img(image)
            # image.show()
            # exit()
            # txt = get_txt(image)
            txt = get_txt_tessocr(image)

            total_ct = total_ct + 1
            correct_ct = correct_ct + (1 if txt.startswith(tag) else 0)
            endtime = time.time()
    print('total ct: %s, correct ct: %s, time cost: %s' % (total_ct, correct_ct, endtime - starttime))
    return correct_ct, total_ct


result_dict = {}
def stat_fs(type):
    correct_ct, total_ct = 0, 0
    for dirpath, dirnames, filenames in os.walk(r'\\192.168.0.110\temp\ocr\%s' % type):
        for dirname in dirnames:
            dir_path = os.path.join(dirpath, dirname)
            tag = dirname[:2]
            # tag_ = tag.replace('C', '0')
            for file in os.listdir(dir_path):
                if not file.split('.')[-1].lower() in ['jpg', 'png']:
                    continue
                img_path = os.path.join(dir_path, file)
                starttime = time.time()
                src_img = ImageOps.invert(Image.open(img_path).convert("RGB"))
                # src_img = Image.open(img_path)
                image = src_img.crop(get_bound(type, src_img))
                image = process_img(image)
                # image.show()
                # exit()
                # txt = get_txt(image)

                txt = get_txt_tessocr(image)

                total_ct = total_ct + 1
                key = tag[1] + ':' + txt[1]
                if key not in result_dict:
                    result_dict[key] = 0
                result_dict[key] = result_dict[key] + 1

                if txt.startswith(tag):
                    correct_ct = correct_ct + (1 if txt.startswith(tag) else 0)
                else:
                    print(tag, txt)
                endtime = time.time()
                # print('total ct: %s, correct ct: %s, time cost: %s' % (total_ct, correct_ct, endtime - starttime))
    return correct_ct, total_ct


# correct_ct, total_ct = stat_ab()
# with open('results.txt', 'a+') as f:
#     f.write('%s:%s,%s,%s\n' % ('奥巴合集', correct_ct, '', total_ct))
# correct_ct, total_ct = stat_fs('富士4450合集1')
# with open('results.txt', 'a+') as f:
#     f.write('%s:%s,%s\n' % ('富士4450合集1', correct_ct, total_ct))
# correct_ct, total_ct = stat_fs('富士4450合集2')
# with open('results.txt', 'a+') as f:
#     f.write('%s:%s,%s\n' % ('富士4450合集2', correct_ct, total_ct))
# correct_ct, total_ct = stat_fs('富士4450合集3')
# with open('results.txt', 'a+') as f:
#     f.write('%s:%s,%s\n' % ('富士4450合集3', correct_ct, total_ct))
# correct_ct, total_ct = stat_fs('富士7000合集')
# with open('results.txt', 'a+') as f:
#     f.write('%s:%s,%s\n' % ('富士7000合集', correct_ct, total_ct))
#     f.write('%s\n' % str(result_dict))
# with open(r'\\192.168.0.110\temp\ocr\奥巴合集\CF-H290I#01.0000000?075380.0125.1536718046\CF-#0413_17_57_40_239297_ocr_img.jpg')


stat_fs('富士4450合集1\1C397G011#001_0_005_2022_04_01_09_03_500-1')





# 奥巴合集: 293 293
# 富士4450合集3：C -> 0  548 548
# 富士4450合集1：C -> 0  854 854
# 富士4450合集2：C -> 0  270 270
# 富士7000合集： C -> 0  517 517

# 奥巴合集: 293, 0, 293
# 富士4450合集1:328,526,854
# 富士4450合集2:205,65,270
# 富士4450合集3:0,548,548
# 富士7000合集:400,117,517

# 型号对应：0:奥巴 只取图像左半边 ||  1:富士4450 只取左下角 || 2:富士7000 只取图像右半边
# 其余型号不进行ORC检测,直接返回

# if endoscope_model in [0]:
#     img = src_img[:, :int(img_w / 2)]
# elif endoscope_model in [1]:
#     img = src_img[int(img_h * 0.4):int(img_h * 0.8), :int(img_w * 0.3)]
# elif endoscope_model in [2]:
#     img = src_img[:, int(img_w / 2):]

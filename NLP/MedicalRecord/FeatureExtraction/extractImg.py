import re
import os
from paddleocr import PaddleOCR, draw_ocr

from Lib.Utils import Utils
from Lib.TextStructral import TextStructral

def extract_text(img_path):
    """
    ocr识别图片，返回文字，每个结果一行
    """
    result = ocr.ocr(img_path, cls=True)
    texts = []
    for line in result:
        texts.append(line[1][0])
        # print(line[1][0])

    return texts

def extract_chaoshen(texts, mrno):
    """
    OCR识别结果，超声后处理
    """
    date, part, jcsj, jcjl  = '', '', '', ''
    jcsj_start, jcjl_start = False, False
    for text in texts:
        if re.match('[报告编号]{3,}[：:](.*)', text):
            r = re.match('[报告编号]{3,}[：:](.*)', text)
            date = r.group(1)[:8]
        elif '检查部位' in text:
            part = text.replace('检查部位', '').replace('：', '').replace(':', '')
        elif '影像描述' in text:
            jcsj_start = True
        elif '影像结论' in text:
            jcjl_start = True
            jcsj_start = False
        elif '检查医生' in text:
            jcjl_start = False
            break
        elif jcsj_start:
            jcsj = jcsj + text
        elif jcjl_start:
            jcjl = jcjl + text

    return '%s,%s,%s,%s,%s,%s' % (mrno, date, 'US', part, jcsj, jcjl)


def extract_fangshe(texts, mrno, type):
    """
    OCR识别结果，放射后处理
    """
    date, part, jcsj, jcjl  = '', '', '', ''
    jcsj_start, jcjl_start = False, False
    for text in texts:
        # if re.match('[A-Z]+号[：:](.*)', text):
        #     r = re.match('[A-Z]+号[：:](.*)', text)
        #     no = r.group(1)
        if '日期' in text:
            date = utils.format_date(text)
        elif '检查名称' in text:
            part = text.replace('检查名称', '').replace('：', '').replace(':', '')
        elif '影像表现' in text:
            jcsj_start = True
        elif '影像诊断' in text:
            jcjl_start = True
            jcsj_start = False
        elif '报告医师' in text:
            jcjl_start = False
            break
        elif jcsj_start:
            jcsj = jcsj + text
        elif jcjl_start:
            jcjl = jcjl + text

    return '%s,%s,%s,%s,%s,%s' % (mrno, date, type, part, jcsj, jcjl)


def extract_bingli(texts, mrno):
    """
    OCR识别结果，病理后处理
    """
    # no, jcsj, jcjl  = '', '', ''
    # jcsj_start, jcjl_start = False, False
    # for text in texts:
    #     if re.match('[病理号]{2,}[：:，](.*)', text):
    #         r = re.match('[病理号]{2,}[：:，](.*)', text)
    #         no = r.group(1)
    #     elif re.search('[肉眼所见：:，]{3,}', text):
    #         print(text, 'match 肉眼所见')
    #         jcsj_start = True
    #     elif re.search('[光镜所见（图）。，]{3,}', text):
    #         print(text, 'match 光镜所见')
    #         jcsj_start = False
    #     elif re.search('[病理诊断：:，]{3,}', text):
    #         print(text, 'match 病理诊断')
    #         jcjl_start = True
    #     elif re.search('[病理科取材医生]{3,}', text):
    #         print(text, 'match 病理科取材医生')
    #         jcjl_start = False
    #         if no != '':
    #             break
    #     elif jcsj_start:
    #         jcsj = jcsj + text
    #     elif jcjl_start:
    #         jcjl = jcjl + text


    result = ts.process_record('\n'.join(texts))

    return '%s,%s,%s,%s' % (mrno, result['送检日期'], result['肉眼所见'].replace('\n', ''), result['病理诊断'].replace('\n', ''))


if __name__ == '__main__':
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

    # 初始化文本结构化类
    ts = TextStructral()
    ts.set_processor()
    ts.load_template(r'data\template\病理.json')

    utils = Utils()

    # 图片处理
    in_folder = r'D:\项目资料\病历辅助诊断\人机大赛数据集\肠道穿孔14'
    # in_folder = r'C:\Users\Administrator\Desktop\新建文件夹'
    results_bl, results_cs, results_fs = [], [], []
    for root, dirs, files in os.walk(in_folder):
        for name in files:
            if name.endswith('.png') or name.endswith('.jpg'):
                file_path = os.path.join(root, name)
                mrno = root.split('\\')[-1]
                # print('mrno: %s' % mrno)
                if not ('病理' in name or '超声' in name or '放射' in name):
                    print('%s not processed!' % file_path)
                    continue

                print('processing %s' % file_path)
                texts = extract_text(file_path)
                if '病理' in name:
                    results_bl.append(extract_bingli(texts, mrno))
                elif '超声' in name:
                    results_cs.append(extract_chaoshen(texts, mrno))
                elif '放射' in name:
                    results_fs.append(extract_fangshe(texts, mrno, name.split('_')[1].replace('.jpg', '').replace('.png', '')))


    # 初始化输出文件
    out_path = r'D:\projects\NLP\MedicalRecord\FeatureExtraction\data\202207人机大赛'
    out_bl = os.path.join(out_path, '病理.txt')
    with open(out_bl, 'a+') as f:
        # f.write('medicalrecordno,流水号,检查所见,诊断意见\n')
        for line in results_bl:
            f.write(line + '\n')

    out_cs = os.path.join(out_path, '超声.txt')
    with open(out_cs, 'a+') as f:
        # f.write('medicalrecordno,流水号,类型,检查部位,检查所见,检查结论\n')
        for line in results_cs:
            f.write(line + '\n')

    out_fs = os.path.join(out_path, '放射.txt')
    with open(out_fs, 'a+') as f:
        # f.write('medicalrecordno,流水号,类型,检查部位,检查所见,检查结论\n')
        for line in results_fs:
            f.write(line + '\n')


#

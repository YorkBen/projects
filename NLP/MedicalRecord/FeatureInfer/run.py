import xlwt
import json
import re
import os
import sys
import argparse

from paddlenlp import Taskflow
from pprint import pprint

sys.path.append('../Lib')
from MRRecordUtil import *
from RegexBase import RegexBase

rutil = RegexBase()

schema = {'症状': ['部位', '否定词', '性质']}
# ie = Taskflow("information_extraction", schema=schema, task_path='./model_ext', device_id=-1) # CPU
ie = Taskflow("information_extraction", schema=schema, task_path='./model_ext', device_id=0)


ner_features = {
    '中腹痛': {'symp': '痛', 'body': '中腹'},
    '上腹痛': {'symp': '痛', 'body': '上腹'},
    '右上腹痛': {'symp': '痛', 'body': '右上腹'},
    '中上腹痛': {'symp': '痛', 'body': '中上腹'},
    '胸骨后疼痛': {'symp': '痛', 'body': '胸骨后'},
    '剑突下疼痛': {'symp': '痛', 'body': '剑突下'},
    '头痛': {'symp': '痛', 'body': '头'},
    '腰痛': {'symp': '痛', 'body': '腰'},
    '剧烈头痛': {'symp': '痛', 'body': '头', 'status': '剧烈'},
    '腹部绞痛': {'symp': '痛', 'body': '腹', 'status': '绞'},
    '餐后腹痛': {'symp': '痛', 'body': '腹', 'status': '餐后'},
    '持续性腹痛': {'symp': '痛', 'body': '腹', 'status': '持续'},
    '放射痛（背部、侧腹）': {'symp': '痛', 'body': ['背', '侧腹'], 'status': '放射'},
    '放射痛（右肩、肩胛和背部）': {'symp': '痛', 'body': ['右肩', '肩胛', '背'], 'status': '放射'},
    '阵发性疼痛': {'symp': '痛', 'status': '阵发'},
    '腹胀': {'symp': '胀', 'body': '腹'},

    '畏寒': {'symp': '畏寒'},
    '恶心': {'symp': '恶心'},
    '呕吐': {'symp': '呕吐'},
    '呕血': {'symp': '呕血'},
    '嗳气': {'symp': '嗳气'},
    '反酸': {'symp': '反酸'},
    '眩晕': {'symp': '眩晕'},
    '腹泻': {'symp': '腹泻'},
    '水肿': {'symp': '水肿'},
    '抽搐': {'symp': '抽搐'}
}


def get_ner_texts(record):
    """
    主诉, 现病史, 病史小结, 首次病程
    """
    texts = []
    texts.append(get_json_value(record, ['入院记录', '主诉']))
    texts.append(get_json_value(record, ['入院记录', '现病史']))

    # text = text + get_json_value(record, ['入院记录', '病史小结', '主诉']) + '。'
    # text = text + get_json_value(record, ['入院记录', '病史小结', '既往史']) + '。'
    # text = text + get_json_value(record, ['入院记录', '病史小结', '查体']) + '。'
    # text = text + get_json_value(record, ['入院记录', '病史小结', '辅助检查']) + '。'
    #
    # text = text + get_json_value(record, ['入院记录', '首次病程', '病例特点', '患者']) + '。'
    # text = text + get_json_value(record, ['入院记录', '首次病程', '病例特点', '现病史']) + '。'
    # text = text + get_json_value(record, ['入院记录', '首次病程', '病例特点', '查体']) + '。'
    # text = text + get_json_value(record, ['入院记录', '首次病程', '病例特点', '辅助检查']) + '。'
    # text = text + get_json_value(record, ['入院记录', '首次病程', '诊断依据']) + '。'
    # text = text + get_json_value(record, ['入院记录', '首次病程', '鉴别诊断']) + '。'

    text = '。'.join(texts)
    texts = rutil.split_text(text)

    return [t.strip() for t in texts if len(t.strip()) > 0]


def match_pattern_texts(patterns, texts):
    """
    规则匹配NER文字
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        for text in texts:
            if pattern in text:
                return True

    return False


def infer_by_feature(symp_text, symp_prob, status_texts, neg_texts, body_texts):
    """
    根据单条文本的NER结果来推理特征。
    """
    print('infer by feature...')
    print('symptom text: %s, symptom prob: %s' % (symp_text, symp_prob))
    print('status texts: %s' % str(status_texts))
    print('neg texts: %s' % str(neg_texts))
    print('body texts: %s' % str(body_texts))

    result = {}
    for k, v in ner_features.items():
        r_val = 2
        if v['symp'] in symp_text:
            if 'body' not in v and 'status' not in v:
                r_val = 1
            elif 'body' not in v and 'status' in v \
                and match_pattern_texts(v['status'], status_texts):
                r_val = 1
            elif 'body' in v and 'status' not in v \
                and match_pattern_texts(v['body'], body_texts):
                r_val = 1
            elif 'body' in v and 'status' in v \
                and match_pattern_texts(v['status'], status_texts) \
                and match_pattern_texts(v['body'], body_texts):
                r_val = 1

            if len(neg_texts) > 0:
                r_val = 0

        result[k] = r_val

    return result


def merge_infer_result(results):
    """
    合并多段文本的推理结果
    """
    result = {}
    for r in results:
        for k, v in r.items():
            if k not in result:
                result[k] = v
            elif result[k] == 2 or v == 1:
                result[k] = v

    return result


def infer_per_text(text):
    """
    [{'症状': [{'end': 13,
          'probability': 0.9998205973689949,
          'relations': {'性质': [{'end': 18,
                                'probability': 0.7157156076722941,
                                'start': 16,
                                'text': '加重'}]},
          'start': 11,
          'text': '呕吐'},
         {'end': 11,
          'probability': 0.9998343057367265,
          'relations': {'性质': [{'end': 18,
                                'probability': 0.6993950986271891,
                                'start': 16,
                                'text': '加重'}]},
          'start': 9,
          'text': '恶心'}]}]
    """
    results = []
    for ie_item in ie(text):
        if '症状' in ie_item:
            for item in ie_item['症状']:
                symp_text = item['text'].replace('疼', '痛')
                symp_prob = item['probability']

                # 性质
                status_texts = []
                if 'relations' in item and '性质' in item['relations']:
                    status_texts = [r_item['text'] for r_item in item['relations']['性质']]

                # 否定词
                neg_texts = []
                if 'relations' in item and '否定词' in item['relations']:
                    neg_texts = [r_item['text'] for r_item in item['relations']['否定词']]

                # 部位
                body_texts = []
                if 'relations' in item and '部位' in item['relations']:
                    body_texts = [r_item['text'] for r_item in item['relations']['部位']]

                r = infer_by_feature(symp_text, symp_prob, status_texts, neg_texts, body_texts)
                results.append(r)

        elif '生理' in ie_item:
            pass

    return merge_infer_result(results)



if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='2409', help='postfix num')
    parser.add_argument('-t', type=str, default='腹痛', help='数据类型')
    args = parser.parse_args()

    postfix = args.p
    data_type = args.t
    if not os.path.exists('../data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    print("postfix: %s, data_type: %s" % (data_type, postfix))
    if not os.path.exists('../data/%s/labeled_ind_%s.txt' % (data_type, postfix)):
        print('mrnos file: ../data/%s/labeled_ind_%s.txt not exists!' % (data_type, postfix))
        exit()

    ## 加载json数据 ############
    keys, json_data = load_data(r'../data/%s/汇总结果_%s.json' % (data_type, postfix), '../data/%s/labeled_ind_%s.txt' % (data_type, postfix))

    ## 写excel ############
    # workbook = load_workbook(r"data/result.xlsx")
    # sheet = workbook["Sheet1"]
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet1")

    ## 写表头 #############
    write_sheet_row(sheet, 0, {k:k for k, v in ner_features.items()})

    ## 写内容 #############
    for ind, item in enumerate(json_data):
        if item is None:
            continue

        item_results = []
        for text in get_ner_texts(item):
            item_results.append(infer_per_text(text))

        item_result = merge_infer_result(item_results)
        write_sheet_row(sheet, ind+1, item_result)


    workbook.save(r'data/%s/NER结果_%s.xlsx' % (data_type, postfix))

    print('saving file to data/%s/NER结果_%s.xlsx' % (data_type, postfix))

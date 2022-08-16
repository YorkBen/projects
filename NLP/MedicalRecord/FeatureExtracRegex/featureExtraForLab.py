# pip install openpyxl
from openpyxl import load_workbook
from openpyxl import Workbook
import json
import re
import os
import sys
import argparse

sys.path.append('../Lib')

from MRRecordUtil import *
from Lib.LabRule import LabRule

def get_json_value(item, key):
    if key not in item:
        return ''
    else:
        return str(item[key])

def load_regex():
    lines = []
    for filename in ['检验_腹痛']:
        with open(r'data/regex/%s.txt' % filename) as f:
            for l in f.readlines():
                lines.append(l.strip())

    return '((' + ')|('.join(lines) + '))'


def get_mzwy_texts(item):
    """
    获取门诊外院中包含实验室数据的文本，以。分割，返回句子数组
    """
    s1 = get_json_value(item['入院记录'], '门诊及院外重要辅助检查') if '入院记录' in item else ''
    s2 = get_json_value(item['入院记录']['病史小结'], '辅助检查') if '入院记录' in item and '病史小结' in item['入院记录'] else ''
    s3 = get_json_value(item['首次病程']['病例特点'], '辅助检查') if '首次病程' in item else ''
    s4 = get_json_value(item['出院记录']['入院情况'], '辅助检查') if '出院记录' in item else ''
    s5 = get_json_value(item['入院记录'], '现病史') if '入院记录' in item else ''

    lr = LabRule()
    regex = load_regex()
    texts = []
    for s in [s1, s2, s3, s4, s5]:
        for t in lr.split_text(s):
            if re.search(regex, t):
                texts.append(t)

    return texts


def merge_result_dict(dict1, dict2):
    """
    合并两个结果字典
    """
    for key in dict1.keys():
        if dict2[key] == 1 or dict1[key] == 2:
            dict1[key] = dict2[key]

    return dict1


def process_texts():
    texts = []
    with open(r'data\腹痛\门诊外院.txt') as f:
        for line in f.readlines():
            texts.append(line)

    results = []
    for line in texts:
        r_dict = lr.get_rule_label_results(lr.process_texts([line]))
        r_dict['文本'] = line
        results.append(r_dict)

    write_sheet_arr_dict(results, r'data\腹痛\实验室数据正则测试_结果.xlsx', 'Sheet1')


def process_records(key_file, json_file, out_path):
    """
    处理记录数据
    """
    keys = load_keys(key_file, with_head=False)
    keys = [e[0] + '_' + e[1] for e in keys]

    json_data = ''
    with open(json_file, encoding='utf-8') as f:
        json_data = json.load(f, strict=False)

    json_data = filter_records(json_data, keys)
    print('Key个数：%s, Json数据个数：%s' % (len(list(keys)), len(json_data)))

    lr = LabRule()
    results = []
    total_ct, illegal_ct = 0, 0
    for item in json_data:
        items_arr = []
        # 检验
        if '检验' in item:
            for item_jy in item['检验']:
                for item_jy_sj in item_jy['数据']:
                    arr = item_jy_sj.split(',')
                    if len(arr) == 6:
                        items_arr.append(arr)
                    else:
                        illegal_ct = illegal_ct + 1
                    total_ct = total_ct + 1
        r_dict_1 = lr.get_rule_label_results(lr.process_strc_items(items_arr))


        # 门诊外院
        texts = get_mzwy_texts(item)
        r_dict_2 = lr.get_rule_label_results(lr.process_texts(texts, debug=True))

        results.append(merge_result_dict(r_dict_1, r_dict_2))
        results.append(r_dict_2)

    write_sheet_arr_dict(results, out_path, 'Sheet1')
    print('Total Count: %s, Illegal Count: %s' % (total_ct, illegal_ct))


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='3456', help='postfix num')
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

    process_records(r'../data/%s/labeled_ind_%s.txt' % (data_type, postfix),
                        r'../data/%s/汇总结果_%s.json' % (data_type, postfix),
                        r'data/%s/实验室正则结果_%s.xlsx' % (data_type, postfix))

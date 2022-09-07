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


def process_records(key_file, json_file, out_path, debug):
    """
    处理记录数据
    """
    keys, json_data = load_data(json_file, key_file)

    lr = LabRule(debug=False)
    results = []
    total_ct, illegal_ct = 0, 0
    for item in json_data:
        items_arr = []
        ## 检验
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
        # print(items_arr)
        # print(r_dict_1['尿液_红细胞_大于_28'])

        ## 门诊外院
        texts = get_mzwy_texts(item, r'../FeatureExtracRegex/data/regex', ['检验'])['检验']
        # for text in texts:
        #     print(text)
        r_dict_2 = lr.get_rule_label_results(lr.process_texts(texts, debug=debug))
        # print(r_dict_2['全血_白细胞_大于_9.5'])


        # ## for debug
        # print(item['医保编号'])
        # print_keys = ['全血_白细胞_大于_10', '血清_总胆红素_大于_22']
        # for dict_c in [r_dict_1, r_dict_2]:
        #     for key, val in dict_c.items():
        #         if key in print_keys:
        #             print(key, dict_c[key])


        results.append(merge_result_dict(r_dict_1, r_dict_2))
        # results.append(r_dict_2)

    # print(results[0])
    write_sheet_arr_dict(results, out_path, 'Sheet1', debug=debug)
    print('Total Count: %s, Illegal Count: %s' % (total_ct, illegal_ct))


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='3456', help='postfix num')
    parser.add_argument('-t', type=str, default='腹痛', help='数据类型')
    parser.add_argument('-d', type=str, default='0', help='调试类型')
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
    debug_type = args.d
    debug = (debug_type != '0')
    labeled_file = r'../data/%s/labeled_ind_%s.txt' % (data_type, postfix) if debug_type != 'label' else r'../data/%s/labeled_ind_%s_debug.txt' % (data_type, postfix)

    process_records(key_file=labeled_file,
                        json_file=r'../data/%s/汇总结果_%s.json' % (data_type, postfix),
                        out_path=r'data/%s/实验室正则结果_%s.xlsx' % (data_type, postfix),
                        debug=debug)

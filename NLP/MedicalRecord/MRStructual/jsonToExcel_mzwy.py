# pip install openpyxl
import json
import re
import os
import sys
import argparse

sys.path.append('../Lib')
sys.path.append('../FeatureExtracRegex/Lib')

from MRRecordUtil import *
from RegexBase import *

rb = RegexBase()

"""
门诊外院
"""

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='2409', help='postfix num')
    parser.add_argument('-t', type=str, default='腹痛', help='数据类型')
    parser.add_argument('-d', type=str, default='0', help='调试类型') # 0，1， label  label的时候是人工调试，不输出excel，只打印信息；1是excel输出调试信息，0是excel不输出调试信息
    args = parser.parse_args()

    postfix = args.p
    data_type = args.t
    if not os.path.exists('../data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    print("postfix: %s, data_type: %s" % (data_type, postfix))
    if not os.path.exists('../data/%s/labeled_ind_%s.txt' % (data_type, postfix)):
        print('mrnos file: data/%s/labeled_ind_%s.txt not exists!' % (data_type, postfix))
        exit()

    debug_type = args.d
    labeled_file = r'../data/%s/labeled_ind_%s.txt' % (data_type, postfix) if debug_type != 'label' else r'../data/%s/labeled_ind_%s_debug.txt' % (data_type, postfix)


    ## 加载json数据 ############
    keys, json_data = load_data(r'../data/%s/汇总结果_%s.json' % (data_type, postfix), labeled_file)

    results = []
    for ind, item in enumerate(json_data):
        result = {
            '医保编号': get_json_value(item, '医保编号'),
            '入院时间': get_json_value(item, '入院时间')
        }
        text_dict = get_mzwy_texts(item, r'../FeatureExtracRegex/data/regex', ['超声', 'CT', 'MR', 'DR', '检验'])
        result.update({key: '\n'.join(val) for key, val in text_dict.items()})

        results.append(result)

    write_sheet_arr_dict(results, r"data/%s/门诊外院文本_%s.xlsx" % (data_type, postfix))

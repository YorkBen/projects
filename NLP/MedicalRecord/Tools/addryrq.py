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


def process_records(key_file, json_file, out_path):
    """
    处理记录数据
    """
    keys, json_data = load_data(json_file, key_file)

    for item in json_data:
        if get_json_value(item, ['入院时间']) == '':
            if get_json_value(item, ['入院记录', '入院时间']) != '':
                item['入院时间'] = get_json_value(item, ['入院记录', '入院时间'])
            elif get_json_value(item, ['出院记录', '入院时间']) != '':
                item['入院时间'] = get_json_value(item, ['出院记录', '入院时间'])

    with open(out_path, "w") as f:
        f.write(json.dumps(json_data, indent=1, separators=(',', ':'), ensure_ascii=False))


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
                    r'../data/%s/汇总结果_%s_r.json' % (data_type, postfix))

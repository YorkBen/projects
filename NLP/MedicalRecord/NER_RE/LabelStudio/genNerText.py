# pip install openpyxl
from openpyxl import load_workbook
import json
import re
import os
import argparse
import sys

sys.path.append('../Lib')

from MRRecordUtil import load_data, load_sheet_dict

def item_to_str(item):
    s = str(item)
    s = s.replace('{', '').replace('}', '').replace('\'', '').replace('"', '')
    return s

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


    keys, json_data = load_data(r'../data/%s/汇总结果_%s.json' % (data_type, postfix), r'../data/%s/labeled_ind_%s.txt' % (data_type, postfix))
    texts = []
    for key, item in zip(keys, json_data):
        if item is None:
            print(key)
            continue

        separator = '\n\n' # '\t\t\t'
        ### 拼接成多行文本
        text = ''
        if '入院记录' in item:
            text = text + separator + item['入院记录']['主诉']
            text = text + separator + item['入院记录']['现病史']
            text = text + separator + item_to_str(item['入院记录']['既往史'])
            text = text + separator + item_to_str(item['入院记录']['手术外伤史'])
            text = text + separator + item_to_str(item['入院记录']['药物过敏史'])
            text = text + separator + item_to_str(item['入院记录']['个人史'])
            text = text + separator + item_to_str(item['入院记录']['婚育史'])
            text = text + separator + item_to_str(item['入院记录']['月经史'])
            text = text + separator + item_to_str(item['入院记录']['家族史'])
            text = text + separator + item_to_str(item['入院记录']['体格检查'])
            text = text + separator + item_to_str(item['入院记录']['专科情况（体检）'])

        for key in ['超声', '放射']:
            if key in item:
                for item_cs in item[key]:
                    for line in item_cs["数据"]:
                        text = text + separator + line

        texts.append({
            "data": {
                "text": text
            }
        })

    if separator == '\t\t\t':
        # 写文本行
        with open(r'data\ner_text.txt', 'w', encoding='utf-8') as f:
            for r in texts:
                f.write('%s\n' % r)
    elif separator == '\n\n':
        # 写json
        with open(r'data\ner_text.json', "w") as f:
            f.write(json.dumps(texts, indent=1, separators=(',', ':'), ensure_ascii=False))

#
#

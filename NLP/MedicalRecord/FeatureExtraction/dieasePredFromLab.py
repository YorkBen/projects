# pip install openpyxl
from openpyxl import load_workbook
from openpyxl import Workbook
import json
import re

from Lib.MRRecordUtil import *
from Lib.LabRule import LabRule


if __name__ == '__main__':
    lr = LabRule()
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

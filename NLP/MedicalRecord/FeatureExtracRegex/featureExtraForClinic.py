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
from Lib.ClinicRule import ClinicRule

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


    cr = ClinicRule(type=data_type, postfix=postfix)

    ## #for debug
    # 设置要提取的特征，如果是所有特征，则注释该行
    # cr.set_proc_features(['ZGYTS1'])

    # results = cr.process(r'../data/腹痛/汇总结果_%s.json' % postfix, r'../data/腹痛/labeled_ind_%s.txt' % postfix)
    results = cr.process(json_file=r'../data/%s/汇总结果_%s.json' % (data_type, postfix),
                         result_file=r'data/%s/临床特征_%s.xlsx' % (data_type, postfix),
                         labeled_file=labeled_file, debug=True)

    ## #for debug
    # for no, item in zip(results['医保编号'], results['子宫压痛']):
    #     print(no, item)

    write_sheet_dict(results, r'data/%s/临床特征_%s.xlsx' % (data_type, postfix), 'Sheet1', debug=debug)
    # manlabeled_data = load_sheet_dict()
    # for key1 in manlabeled_data.keys():
    #     for key2 in manlabeled_data[key1].keys():
    #         if manlabeled_data[key1][key2] == '':
    #             manlabeled_data[key1][key2] = 0
    #         else:
    #             manlabeled_data[key1][key2] = int(manlabeled_data[key1][key2])

    # results = cr.stat_and_tag_results(results, manlabeled_data)
    # cr.write_to_excel(results, r'../data/%s/f_%s.xlsx' % (type, postfix), debug=True)

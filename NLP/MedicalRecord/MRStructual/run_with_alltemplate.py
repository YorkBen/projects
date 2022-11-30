import re
import os
import sys
import json
import time
import datetime
import numpy as np
import argparse
import logging

sys.path.append('../Lib')

from Lib.TextStructral import TextStructral
from Lib.CommonSplit import CommonSplit
from FileUtil import load_file, load_dict
from RegexUtil import RegexUtil
from MRRecordUtil import *

logging.basicConfig(level=logging.INFO)
utils = RegexUtil()


def init_with_mrno(file_path, with_head=True, sperator='	'):
    """
    输入：mrno	入院日期(20220101)
    初始化病历数据
    """
    results = {}
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head or line.strip() == '':
                continue

            # arr = line.strip().split(sperator)
            # mr_no, bed_no = arr[0], arr[1]
            # if len(arr) != 2:
            #     print('illegal line: %s' % line)
            #
            # results['%s' % (mr_no, bed_no)] = {
            #     '住院号': mr_no,
            #     '床号': bed_no
            # }

            mr_no = line.strip()
            results[mr_no] = {}

    return results


def get_mr_no(line):
    """
    获取医保号码
    """
    return line.split('||')[0]


def join_others(data_records, o_dict, name, delta=90, fix_date=False):
    """
    加载实验室数据、超声、放射、病理、医嘱
    """
    # o_dict = load_others_dict('chaoshen')
    print('开始处理%s....' % name)
    starttime = time.time()
    has_o_num = 0
    for r_rd in data_records:
        mr_no = get_json_value(r_rd, ['入院记录', '一般信息', '住院号'])
        print(mr_no)
        if mr_no in o_dict:
            has_o_num = has_o_num + 1
            o_records = o_dict[mr_no]
            o_date_list = []
            for r_o in o_records:
                o_date_list.append(r_o[0])
            o_date_list = sorted(list(set(o_date_list)))

            # # 补齐日期中的','
            # if fix_date:
            #     ih_date = utils.fix_datestr(ih_date, o_date_list[0], ',')
            #     r_rd['入院记录']['入院时间'] = ih_date

            # ih_date_int = int(ih_date)
            # 附加结果
            r_rd[name] = []

            # 按时间来规整
            # out_date1 = get_json_value(r_rd, ['入院记录', '一般信息', '入院时间'])
            # out_date2 = get_json_value(r_rd, ['出院记录', '一般信息', '入院时间'])
            # out_date = out_date1 if out_date1 !='' else out_date2
            for o_date in o_date_list:
                # if check_date(out_date, o_date, delta):
                date_result = {
                    '日期': o_date,
                    '数据': []
                }
                for r_insp in o_records:
                    if r_insp[0] == o_date:
                        date_result['数据'].append(','.join(r_insp[1:]))
                r_rd[name].append(date_result)

    endtime = time.time()
    print('%s处理记录数: ' % name, has_o_num)
    print('%s处理用时: ' % name, endtime - starttime)
    return data_records


def check_date(date1, date2, delta):
    """
    比较时间和入院时间：
    1. 如果入院时间为空，则直接返回True。
    2. 入院时间不为空，比较时间为空，返回False。
    3. 两个时间都不为空，看时间差值绝对值是否小于delta，是则True，否则False
    """
    if date1 == '':
        return True
    elif date2 == '':
        return True
    else:
        try:
            date1_dt = datetime.datetime.strptime(date1, '%Y%m%d').date()
            date2_dt = datetime.datetime.strptime(date2, '%Y%m%d').date()
            if abs((date1_dt - date2_dt).days) < delta:
                return True
            else:
                return False
        except:
            return False


def stat_empty(lbl_dict):
    """
    统计没有匹配数
    """
    empty_ct, total_ct = 0, 0
    for key in lbl_dict:
        if lbl_dict[key] == '':
            print(key)
            empty_ct = empty_ct + 1
        total_ct = total_ct + 1

    print('%s/%s' % (empty_ct, total_ct))


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='all', help='postfix num')
    parser.add_argument('-p2', type=str, default='1', help='postfix num')
    parser.add_argument('-t', type=str, default='功能性肠胃病', help='数据类型')
    args = parser.parse_args()

    postfix = args.p
    postfix2 = args.p2
    data_type = args.t
    if not os.path.exists('data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    if not os.path.exists('../data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    print("postfix: %s, data_type: %s" % (data_type, postfix))
    if not os.path.exists('data/%s/labeled_ind_%s.txt' % (data_type, postfix)):
        print('mrnos file: data/%s/labeled_ind_%s.txt not exists!' % (data_type, postfix))
        exit()


    print('根据医保编号数据初始化病历对象...')
    result_records = init_with_mrno(file_path=r'data/%s/labeled_ind/labeled_ind_%s' % (data_type, postfix2), with_head=False, sperator=' ')
    keys = list(result_records.keys())

    ts = TextStructral()
    ps = CommonSplit(template_file=r'data/template/all.json')
    ############入院记录######################################
    starttime = time.time()
    print('开始处理入院记录....')
    records = ts.load_records('data/%s/tmp/mr_ry_%s.txt' % (data_type, postfix), keys)
    for idx, record in enumerate(records):
        if len(record) < 2 or len(record[1]) < 100:
            continue
        result = ps.process(record[1], "root")
        if '一般信息' in result['root']:
            if '住院号' in result['root']['一般信息']:
                mr_no = result['root']['一般信息']['住院号']
            else:
                mr_no = get_mr_no(record[0])
            key = mr_no
            print(idx, key)
            if key in result_records:
                if '入院记录' in result_records[key]:
                    result_records[key]['入院记录'] = ps.update_val(result_records[key]['入院记录'], result['root'])
                else:
                    result_records[key]['入院记录'] = result['root']

                if get_json_value(result_records[key], ['入院记录', '一般信息', '住院号']) == '':
                    result_records[key]['入院记录']['一般信息']['住院号'] = key
            else:
                print('%s not in result_records' % key)

    endtime = time.time()
    print('入院记录处理记录数: ', len(records))
    print('入院记录处理用时: ', endtime - starttime)


    ##############首次病程#######################################
    starttime = time.time()
    print('开始处理首次病程....')
    records = ts.load_records('data/%s/tmp/mr_sc_%s.txt' % (data_type, postfix), keys)
    for idx, record in enumerate(records):
        if len(record) < 2 or len(record[1]) < 100:
            continue
        result = ps.process(record[1], "首次病程记录")
        if '首次病程记录' in result:
            if '住院号' in result['首次病程记录']:
                mr_no = result['首次病程记录']['住院号']
            else:
                mr_no = get_mr_no(record[0])
            key = mr_no
            print(idx, key)
            if key in result_records:
                if '首次病程记录' in result_records[key]:
                    result_records[key]['首次病程记录'] = ps.update_val(result_records[key]['首次病程记录'], result['首次病程记录'])
                else:
                    result_records[key]['首次病程记录'] = result['首次病程记录']
            else:
                print('%s not in result_records' % key)

    endtime = time.time()
    print('首次病程处理记录数: ', len(records))
    print('首次病程处理用时: ', endtime - starttime)


    ##############日常病程#######################################
    starttime = time.time()
    print('开始处理日常病程记录....')
    records = ts.load_records('data/%s/tmp/mr_rc_%s.txt' % (data_type, postfix), keys)
    for idx, record in enumerate(records):
        if len(record) < 2 or len(record[1]) < 100:
            continue

        result = ps.process(record[1], "日常病程记录")
        if '日常病程记录' in result:
            if '住院号' in result['日常病程记录']:
                mr_no = result['日常病程记录']['住院号']
            else:
                mr_no = get_mr_no(record[0])
            key = mr_no
            print(idx, key)
            if key in result_records:
                if not '日常病程记录' in result_records[key]:
                    result_records[key]['日常病程记录'] = []
                result_records[key]['日常病程记录'].append(result['日常病程记录'])
            else:
                print('%s not in result_records' % key)

    endtime = time.time()
    print('日常病程处理记录数: ', len(records))
    print('日常病程处理用时: ', endtime - starttime)


    ##############出院记录#######################################
    starttime = time.time()
    print('开始处理出院记录....')
    records = ts.load_records('data/%s/tmp/mr_cy_%s.txt' % (data_type, postfix), keys)
    for idx, record in enumerate(records):
        if len(record) < 2 or len(record[1]) < 100:
            continue
        result = ps.process(record[1], "root")
        if '一般信息' in result['root']:
            if '住院号' in result['root']['一般信息']:
                mr_no = result['root']['一般信息']['住院号']
            else:
                mr_no = get_mr_no(record[0])
            key = mr_no
            print(idx, key)
            if key in result_records:
                if '出院记录' in result_records[key]:
                    result_records[key]['出院记录'] = ps.update_val(result_records[key]['出院记录'], result['root'])
                else:
                    result_records[key]['出院记录'] = result['root']
            else:
                print('%s not in result_records' % key)

    endtime = time.time()
    print('出院记录处理记录数: ', len(records))
    print('出院记录处理用时: ', endtime - starttime)


    # ##############出院诊断证明#######################################
    # starttime = time.time()
    # print('开始处理出院诊断证明书....')
    # records = ts.load_records('data/records/出院诊断证明书.txt')
    # ts.load_template('data/template/出院诊断证明书.json')
    # ts.set_processor('label')
    # results = ts.process()
    # ct = 0
    # for i in range(len(records)):
    #     mr_no = get_mr_no(records[i])
    #     if mr_no is not None and mr_no in result_records \
    #         and check_date(result_records[mr_no]['入院时间'], results[i]['入院时间'], 300):
    #         ct = ct + 1
    #         result_records[mr_no]['出院诊断证明书'] = results[i]
    # endtime = time.time()
    # print('出院诊断证明书记录数: ', ct)
    # print('出院诊断证明书处理用时: ', endtime - starttime)


    ###############结果转换为数组###################################
    # 日常病程排序
    # results = []
    # for key in result_records:
    #     if '日常病程' in result_records[key]:
    #         result_records[key]['日常病程'] = sorted(result_records[key]['日常病程'], key=lambda x: x['DATE'])
    #     results.append(result_records[key])


    ## 下面数据没有床号，只能根据住院号拼接
    results = [val for key, val in result_records.items()]

    #########白细胞计数、中性粒细胞计数##############################
    # results = join_inspec(results)
    results = join_others(results, load_dict(r"data/%s/tmp/is_%s.txt" % (data_type, postfix)), "检验", fix_date=True)
    results = join_others(results, load_dict(r'data/%s/tmp/fangshe_%s.txt' % (data_type, postfix)), '放射')
    results = join_others(results, load_dict(r'data/%s/tmp/yizhu_%s.txt' % (data_type, postfix)), '医嘱')
    results = join_others(results, load_dict(r"data/%s/tmp/bingli_%s.txt" % (data_type, postfix)), '病理')

    ###############结果写到文件####################################
    ts.write_result('../data/%s/汇总结果_%s.json' % (data_type, postfix2), results)

    # stat_empty(lbl_dict)

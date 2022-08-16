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
from FileUtil import load_file, load_dict
from RegexUtil import RegexUtil

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

            arr = line.strip().split(sperator)
            mr_no, ry_date = arr[0], ''
            if len(arr) == 2:
                ry_date = arr[1]
                if ry_date != '' and not re.match('[0-9]{8}', ry_date):
                    print('%s:%s line illegal!' % (idx, line))
                    exit()

            results[mr_no] = {
                '医保编号': mr_no,
                '入院时间': ry_date
            }

    return results


def get_mr_no(record):
    """
    获取医保号码
    """
    for line in record:
        if '||' in line:
            return line.split('||')[0]


# def load_choose_map(file_path):
#     """
#     加载选择数据集作为字典
#     """
#     lbl_lines = load_file(file_path, separator='	')
#
#     lbl_dict = {}
#     for line in lbl_lines:
#         key = ''.join(line).replace(',', '，').replace(' ', '').replace('\"\"', '\"')
#         # key = line[0]
#         lbl_dict[key] = ''
#
#     return lbl_dict


# def join_inspec(data_records):
#     """
#     附加实验室检验数据
#     """
#     dict_insp = load_insp_dict()
#     starttime = time.time()
#     print('开始处理实验室检验数据....')
#     has_insp_num = 0
#     for r_rd in data_records:
#         ih_date = r_rd['入院记录']['入院时间']
#         mr_no = r_rd['医保编号']
#         if ih_date != '' and mr_no in dict_insp:
#             has_insp_num = has_insp_num + 1
#             insp_records = dict_insp[mr_no]
#             insp_date_list = []
#             for r_insp in insp_records:
#                 insp_date_list.append(r_insp[0])
#             insp_date_list = sorted(list(set(insp_date_list)))
#             # 补齐日期中的','
#             ih_date = utils.fix_datestr(ih_date, insp_date_list[0], ',')
#             r_rd['入院记录']['入院时间'] = ih_date
#             ih_date_int = int(ih_date)
#             # 附加结果
#             r_rd['实验室数据'] = []
#
#             # 按时间来规整
#             for insp_date in insp_date_list:
#                 insp_date_int = int(insp_date)
#                 if abs(insp_date_int - ih_date_int) < 90:
#                     date_result = {
#                         '检验日期': insp_date,
#                         '检验数据': []
#                     }
#                     for r_insp in insp_records:
#                         if r_insp[0] == insp_date:
#                             date_result['检验数据'].append(','.join(r_insp[1:]))
#                     r_rd['实验室数据'].append(date_result)
#
#     endtime = time.time()
#     print('实验室数据处理记录数: ', has_insp_num)
#     print('实验室数据处理用时: ', endtime - starttime)
#     return data_records


def join_others(data_records, o_dict, name, delta=90, fix_date=False):
    """
    加载实验室数据、超声、放射、病理、医嘱
    """
    # o_dict = load_others_dict('chaoshen')
    print('开始处理%s....' % name)
    starttime = time.time()
    has_o_num = 0
    for r_rd in data_records:
        if r_rd['医保编号'] in o_dict:
            has_o_num = has_o_num + 1
            o_records = o_dict[r_rd['医保编号']]
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
            for o_date in o_date_list:
                if check_date(r_rd['入院时间'], o_date, delta):
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
        return False
    else:
        date1_dt = datetime.datetime.strptime(date1, '%Y%m%d').date()
        date2_dt = datetime.datetime.strptime(date2, '%Y%m%d').date()
        if abs((date1_dt - date2_dt).days) < delta:
            return True
        else:
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
    parser.add_argument('-p', type=str, default='2409', help='postfix num')
    parser.add_argument('-t', type=str, default='腹痛', help='数据类型')
    args = parser.parse_args()

    postfix = args.p
    data_type = args.t
    if not os.path.exists('data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    if not os.path.exists('../data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    print("postfix: %s, data_type: %s" % (data_type, postfix))
    if not os.path.exists('../data/%s/labeled_ind_%s.txt' % (data_type, postfix)):
        print('mrnos file: ../data/%s/labeled_ind_%s.txt not exists!' % (data_type, postfix))
        exit()


    print('根据医保编号数据初始化病历对象...')
    result_records = init_with_mrno(file_path=r'../data/%s/labeled_ind_%s.txt' % (data_type, postfix), with_head=False, sperator='	')

    ts = TextStructral()
    ############入院记录######################################
    starttime = time.time()
    print('开始处理入院记录....')
    records = ts.load_records('data/%s/tmp/mr_ry_%s.txt' % (data_type, postfix))
    ts.load_template('data/template/入院记录.json')
    ts.set_processor()
    results = ts.process()
    # print(json.dumps(results, indent=1, separators=(',', ':'), ensure_ascii=False))
    # ts.write_result('data/入院记录.json')

    # 处理病史小结
    # ts.set_processor('label')
    # ts.load_template('data/template/病史小结.json')
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        if mr_no is not None and mr_no in result_records \
            and check_date(result_records[mr_no]['入院时间'], results[i]['入院时间'], 10):

            # 对现病史和病史小结数据做处理
            results[i]['现病史'] = results[i]['现病史'].replace(',', '，').replace(' ', '').replace('\"\"', '\"').replace('\n', '').replace('\t', '')
            # results[i]['病史小结'] = ts.process_record('病史小结：' + results[i]['病史小结'])
            result_records[mr_no]['入院记录'] = results[i]
            ct = ct + 1

    endtime = time.time()
    print('入院记录处理记录数: ', ct)
    print('入院记录处理用时: ', endtime - starttime)


    ##############首次病程#######################################
    starttime = time.time()
    print('开始处理首次病程....')
    records = ts.load_records('data/%s/tmp/mr_sc_%s.txt' % (data_type, postfix))
    ts.load_template('data/template/首次病程.json')
    ts.set_processor('label')
    results = ts.process()
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        if mr_no is not None and mr_no in result_records \
            and check_date(result_records[mr_no]['入院时间'], results[i]['DATE'], 10):
            ct = ct + 1
            result_records[mr_no]['首次病程'] = results[i]
    endtime = time.time()
    print('首次病程处理记录数: ', ct)
    print('首次病程处理用时: ', endtime - starttime)


    ##############日常病程#######################################
    starttime = time.time()
    print('开始处理日常病程记录....')
    records = ts.load_records('data/%s/tmp/mr_rc_%s.txt' % (data_type, postfix))
    ts.load_template('data/template/日常病程.json')
    ts.set_processor()
    results = ts.process()
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        if mr_no is not None and mr_no in result_records \
            and check_date(result_records[mr_no]['入院时间'], results[i]['DATE'], 90):
            ct = ct + 1
            if '日常病程' not in result_records[mr_no]:
                result_records[mr_no]['日常病程'] = []
            result_records[mr_no]['日常病程'].append(results[i])

    endtime = time.time()
    print('日常病程处理记录数: ', ct)
    print('日常病程处理用时: ', endtime - starttime)


    ##############出院记录#######################################
    starttime = time.time()
    print('开始处理出院记录....')
    records = ts.load_records('data/%s/tmp/mr_cy_%s.txt' % (data_type, postfix))
    ts.load_template('data/template/出院记录.json')
    ts.set_processor('label')
    results = ts.process()
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        if mr_no is not None and mr_no in result_records \
            and check_date(result_records[mr_no]['入院时间'], results[i]['入院时间'], 10):
            ct = ct + 1
            result_records[mr_no]['出院记录'] = results[i]
    endtime = time.time()
    print('出院记录处理记录数: ', ct)
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
    results = []
    for key in result_records:
        if '日常病程' in result_records[key]:
            result_records[key]['日常病程'] = sorted(result_records[key]['日常病程'], key=lambda x: x['DATE'])
        results.append(result_records[key])


    #########白细胞计数、中性粒细胞计数##############################
    # results = join_inspec(results)
    results = join_others(results, load_dict(r"data/%s/tmp/is_%s.txt" % (data_type, postfix)), "检验", fix_date=True)
    results = join_others(results, load_dict(r'data/%s/tmp/chaoshen_%s.txt' % (data_type, postfix)), '超声')
    results = join_others(results, load_dict(r'data/%s/tmp/fangshe_%s.txt' % (data_type, postfix)), '放射')
    # results = join_others(results, load_dict(r'data/%s/tmp/bingli_%s.txt' % (data_type, postfix)), '病理', delta=360)
    results = join_others(results, load_dict(r"data/%s/tmp/yz_%s.txt" % (data_type, postfix)), '医嘱')

    ###############结果写到文件####################################
    ts.write_result('../data/%s/汇总结果_%s.json' % (data_type, postfix), results)

    # stat_empty(lbl_dict)

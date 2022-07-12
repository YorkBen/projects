import time
import logging
import json
import numpy as np
from Lib.TextStructral import TextStructral
from Lib.FileUtil import load_file, load_dict
from Lib.Utils import Utils

logging.basicConfig(level=logging.INFO)
utils = Utils()

def get_mr_no(record):
    """
    获取医保号码
    """
    for line in record:
        if '||' in line:
            return line.split('||')[0]


def load_choose_map(file_path):
    """
    加载选择数据集作为字典
    """
    lbl_lines = load_file(file_path, separator='	')

    lbl_dict = {}
    for line in lbl_lines:
        key = ''.join(line).replace(',', '，').replace(' ', '').replace('\"\"', '\"')
        # key = line[0]
        lbl_dict[key] = ''

    return lbl_dict


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
        ih_date = r_rd['入院记录']['入院时间']
        mr_no = r_rd['医保编号']
        if ih_date != '' and mr_no in o_dict:
            has_o_num = has_o_num + 1
            o_records = o_dict[mr_no]
            o_date_list = []
            for r_o in o_records:
                o_date_list.append(r_o[0])
            o_date_list = sorted(list(set(o_date_list)))
            # 补齐日期中的','
            if fix_date:
                ih_date = utils.fix_datestr(ih_date, o_date_list[0], ',')
                r_rd['入院记录']['入院时间'] = ih_date

            ih_date_int = int(ih_date)
            # 附加结果
            r_rd[name] = []

            # 按时间来规整
            for o_date in o_date_list:
                o_date_int = int(o_date)
                if abs(o_date_int - ih_date_int) < delta:
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


if __name__ == "__main__":
    ts = TextStructral()
    postfix = '1432'

    # 因为数据总量6万条，这是从中筛选的部分
    # # 原始2049条数据
    # 'data/labeled_ind_2409\.txt'
    # 202207根据主诉补充新增1432条数据
    lbl_dict = load_choose_map(r'data/labeled_ind_%s.txt' % postfix)
    print(len(lbl_dict))

    ############入院记录######################################
    starttime = time.time()
    print('开始处理入院记录....')
    records = ts.load_records('data/records/入院记录_%s.txt' % postfix)
    ts.load_template('data/template/入院.json')
    ts.set_processor()
    results = ts.process()
    # print(json.dumps(results, indent=1, separators=(',', ':'), ensure_ascii=False))
    ts.write_result('data/入院记录.json')

    # 处理病史小结
    ts.set_processor('label')
    ts.load_template('data/template/病史小结.json')
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        if mr_no is None:
            mr_no = ''

        # 对现病史和病史小结数据做处理
        results[i]['现病史'] = results[i]['现病史'].replace(',', '，').replace(' ', '').replace('\"\"', '\"').replace('\n', '').replace('\t', '')
        results[i]['病史小结'] = ts.process_record('病史小结：' + results[i]['病史小结'])
        key = mr_no + results[i]['现病史']

        # 比较key不同
        if mr_no == '10020133':
            print(key)

        if key in lbl_dict:
            ct = ct + 1
            lbl_dict[key] = {
                '医保编号': mr_no,
                '入院记录': results[i]
            }
        else:
            print(key)
    print(ct)

    result_records = {}
    for key in lbl_dict.keys():
        # 比较key不同
        if '医保编号' not in lbl_dict[key]:
            print(key)
        result_records[lbl_dict[key]['医保编号']] = lbl_dict[key]
    endtime = time.time()
    print('入院记录处理记录数: ', len(result_records))
    print('入院记录处理用时: ', endtime - starttime)


    ##############首次病程#######################################
    starttime = time.time()
    print('开始处理首次病程....')
    records = ts.load_records('data/records/首次病程.txt')
    ts.load_template('data/template/首次病程.json')
    ts.set_processor('label')
    results = ts.process()
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        mr_no = '' if mr_no is None else mr_no
        if mr_no in result_records and results[i]['DATE'] != '':
            date1 = int(result_records[mr_no]['入院记录']['入院时间'])
            date2 = int(results[i]['DATE'])
            if date2 >= date1 and date2 - date1 < 10:
                ct = ct + 1
                result_records[mr_no]['首次病程'] = results[i]
    endtime = time.time()
    print('首次病程处理记录数: ', ct)
    print('首次病程处理用时: ', endtime - starttime)


    ##############日常病程#######################################
    starttime = time.time()
    print('开始处理日常病程记录....')
    records = ts.load_records('data/records/日常病程记录.txt')
    ts.load_template('data/template/日常病程.json')
    ts.set_processor()
    results = ts.process()
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        mr_no = '' if mr_no is None else mr_no
        if mr_no in result_records and results[i]['DATE'] != '':
            date1 = int(result_records[mr_no]['入院记录']['入院时间'])
            date2 = int(results[i]['DATE'])
            if date2 >= date1 and date2 - date1 < 90:
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
    records = ts.load_records('data/records/出院记录.txt')
    ts.load_template('data/template/出院记录.json')
    ts.set_processor('label')
    results = ts.process()
    ct = 0
    for i in range(len(records)):
        mr_no = get_mr_no(records[i])
        mr_no = '' if mr_no is None else mr_no
        if mr_no in result_records and results[i]['出院时间'] != '':
            date1 = int(result_records[mr_no]['入院记录']['入院时间'])
            date2 = int(results[i]['出院时间'])
            if date2 >= date1 and date2 - date1 < 300:
                ct = ct + 1
                result_records[mr_no]['出院记录'] = results[i]
    endtime = time.time()
    print('出院记录处理记录数: ', ct)
    print('出院记录处理用时: ', endtime - starttime)


    ###############结果转换为数组###################################
    # 日常病程排序
    results = []
    for key in result_records:
        if '日常病程' in result_records[key]:
            result_records[key]['日常病程'] = sorted(result_records[key]['日常病程'], key=lambda x: x['DATE'])
        results.append(result_records[key])


    #########白细胞计数、中性粒细胞计数##############################
    # results = join_inspec(results)
    results = join_others(results, load_dict(r"data/tmp/is_all_%s.txt" % postfix), "检验", fix_date=True)
    results = join_others(results, load_dict(r'data/tmp/chaoshen_%s.txt' % postfix), '超声')
    results = join_others(results, load_dict(r'data/tmp/fangshe_%s.txt' % postfix), '放射')
    results = join_others(results, load_dict(r'data/tmp/bingli_%s.txt' % postfix), '病理', delta=360)
    results = join_others(results, load_dict(r"data/tmp/yz_all_%s.txt" % postfix), '医嘱')

    ###############结果写到文件####################################
    ts.write_result('data/汇总结果_%s.json' % postfix, results)

    # stat_empty(lbl_dict)

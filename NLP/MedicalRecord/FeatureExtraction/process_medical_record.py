import os
import re
from utils import *

data_path = r"data/medical_record.csv"
# data_path = r"data/medical_record_test.csv"

"""
医疗记录数据提取
"""

def find_temp_pattern(str):
    # 体温： 36.6℃
    pattern = re.compile(r'体温[：: ]?([34]\d\.?\d?)℃')
    for match in re.finditer(pattern, str):
        return match.group(1)


def find_date_pattern(str):
    # 日期
    pattern = re.compile('(20[12]\d)[年\-\/]([01]?\d)[月\-\/]([0123]?\d)')
    for match in re.finditer(pattern, str):
        # return match.group(1) + '-' + match.group(2) + '-' + match.group(3)
        part1 = match.group(1)
        part2 = match.group(2) if len (match.group(2)) == 2 else '0' + match.group(2)
        part3 = match.group(3) if len (match.group(2)) == 2 else '0' + match.group(2)
        # part1 = part1.replace(',', '1')
        # part2 = part2.replace(',', '1')
        # part3 = part3.replace(',', '1')

        return part1 + part2 + part3


def find_nowdiease_pattern(str):
    res = re.match('^现病史[：:]?(.*)|^.*? 2、现病史[：:]?(.*)', str)
    if res is not None:
        return res.group(1)
    else:
        return None


def check_pastdiease_pattern(str):
    res = re.match('.*既往史.*', str)
    if res is not None:
        return True
    else:
        return False


def find_ftt_pattern(str):
    res1 = re.match('^腹部.*反跳痛[:：]?([有无]).*', str)
    if res1 is not None:
        if res1.group(1) == '无':
            return '无'
        elif res1.group(1) == '有':
            return '有'
            # res2 = re.match('^腹部.*反跳痛[:：]?有部位[:：]?(.*?)肝.*', str)
            # if res2 is None:
            #     return '有'
            # else:
            #     word = res2.group(1).strip().replace('反跳痛', '').replace('。', '')
            #     if word == '':
            #         return '有'
            #     else:
            #         if ('剑突' not in word and '腹' not in word and '麦氏点' not in word and '脐' not in word) or len(word) > 10:
            #         # if word not in ['右腹', '左腹', '左下腹', '左上腹', '右下腹', '右上腹', '全腹', '右下腹部',
            #         #                 '右上腹部', '上腹部', '下腹部', '麦氏点', '右上腹为主', '中下腹部', '全腹部',
            #         #                 '脐周', '', '', '', '']:
            #             print(str)
            #             print(word)
            #         return word
    else:
        return None


def find_yuejing_zhouqi_pattern(sentence):
    """
    提取月经周期
    """
    search_obj1 = re.search(r'月经.*周期([2-9][0-9])天', sentence)
    if search_obj1 is not None:
        # print(search_obj1[1], 1)
        return search_obj1[1]

    search_obj2 = re.search(r'月经规[律则].*/[0-9-]*([2-9][0-9])', sentence)
    if search_obj2 is not None:
        # print(search_obj2[1], 2)
        return search_obj2[1]

    search_obj3 = re.search(r'月经.*/[0-9-]*([2-9][0-9])', sentence)
    if search_obj3 is not None:
        # print(search_obj3[1], 3)
        return search_obj3[1]

    return ''


def grep_pattern_in_raw(pattern):
    # 在原始数据中抓模式
    with open(data_path) as f:
        for line in f.readlines():
            if pattern in line:
                print(line)


def record_split(record_type=None):
    """
    切分医疗记录，同一个入院记录的文本行合并，只输出入院记录
    """
    medical_records, mr_item, mr_cnt = [], [], ''
    with open(data_path) as f:
        for idx, line in enumerate(f.readlines()):
            # line = line.strip()
            if idx > 0:
                # 还没有进入病历内部
                if len(mr_item) == 0:
                    elems = line.split(',')
                    # 找到病历第一行
                    if len(elems) == 5 and '\"' in elems[4]:
                        mr_item = elems[0:4]
                        mr_cnt = elems[4].replace('\"', '')
                elif line.strip() != '':
                    # 到达一个病历的结束行
                    # 保留\n在后面作为行分割使用
                    if line.strip().endswith('\"'):
                        # 只记录入院记录
                        if record_type and record_type in mr_item[3] or not record_type:
                            mr_item.append(mr_cnt + line[:-1])
                            medical_records.append(mr_item)
                        # 清空数据，初始化
                        mr_item = []
                        mr_cnt = ''
                    else:
                        mr_cnt = mr_cnt + line

            # if len(medical_records) == 2:
            #     break

    return medical_records


def choose_records(data_records, labeled_file):
    """
    从所有病历数据中选出待标记的数据
    病历数据：
        输入：data_records
        格式：'medicalrecordno', '###', '###', '###', '病历文本', '入院日期', '现病史'
    标记数据：
        输入：'data/labeled_ind.txt'
        格式：'medicalrecordno', '现病史'
    输出：
        格式：'medicalrecordno', '###', '###', '###', '病历文本', '入院日期', '现病史'
    """
    lbl_lines = load_file(labeled_file, separator='	')
    lbl_dict = {}
    for line in lbl_lines:
        key = ''.join(line).replace(',', '，').replace(' ', '')
        lbl_dict[key] = ''

    for line in data_records:
        key = line[0] + line[6]
        if key in lbl_dict:
            lbl_dict[key] = line

    results = []
    empty_ct = 0
    for key in lbl_dict:
        val = lbl_dict[key]
        if val == '':
            empty_ct = empty_ct + 1
        results.append(val)

    print('%s empty records' % empty_ct)

    return results


def extract_temp(results):
    # 抽取体温
    for record in results:
        temp_records = []
        r_lines = record[4].split('\n')
        for ind, line in enumerate(r_lines):
            if line is not None and '体温' in line and '℃' in line:
                temp = find_temp_pattern(line)
                if temp is None:
                    continue
                if '生命体征' in line or '一般测量' in line or '一般情况' in line:
                    prior = 1
                elif '查体' in line:
                    prior = 2
                elif '脉搏' in line:
                    prior = 3
                else:
                    prior = 4
                temp_records.append((temp, prior))

        # print(temp_records)
        temp_records = sorted(temp_records, key=lambda x: x[1])
        if len(temp_records) > 0:
            if float(temp_records[0][0]) > 37.5:
                record.append(1)
            else:
                record.append(0)
        else:
            # record.append(-1)
            record.append(2)

    return results


def extract_date(results):
    # 抽取第一个日期
    date = ''
    for record in results:
        for line in record[4].split('\n'):
            date = find_date_pattern(line)
            # print(line)
            # print(date)
            if date is not None:
                break

        if date is None:
            date = ''
        record.append(date)

    return results


def extract_now_dieasehistory(results):
    # 抽取现病史
    history = ''
    for record in results:
        r_lines = record[4].split('\n')
        for ind, line in enumerate(r_lines):
            history = find_nowdiease_pattern(line)

            if history is not None:
                history = history.strip()
                i = 1
                while not check_pastdiease_pattern(r_lines[ind + i]):
                    history = history + r_lines[ind + i].strip()
                    i = i + 1
                break

            # print(line)
            # print(history)

        if history is None:
            history = ''
        record.append(history.replace(' ', '').replace('\"\"', '\"').replace(',', '，'))

    return results


def extract_ftt(results):
    # 抽取反跳痛特征
    history = ''
    for record in results:
        r_lines = record[4].split('\n')
        for ind, line in enumerate(r_lines):
            ftt = find_ftt_pattern(line)
            # print(line)
            # print(ftt)
            if ftt is not None:
                break

        if ftt is None:
            # record.append(-1)
            record.append(2)
        elif ftt == '无':
            record.append(0)
        else:
            record.append(1)

    return results


def extract_jjj(results):
    # 抽取肌紧张特征
    history = ''
    hint_re = '肌紧张'
    for record in results:
        r_list, segs, pres, posts = findPosNegNonePattern(record[4].split('\n'), hint_re)
        r_s = set(r_list)
        if 1 in r_s:
            record.append(1)
        elif 0 in r_s:
            record.append(0)
        else:
            record.append(2)

    return results


def extract_tj(results):
    """
    提取停经特征
    """
    for record in results:
        # record[6] 现病史, record[5] 入院日期
        zq = find_yuejing_zhouqi_pattern(record[6])
        d1 = findLabelDateValuePattern(record[6], '末次月经|lmp')
        d2 = findLabelDateValuePattern(record[6], 'pmp')

        # 根据pmp补全末次月经日期或者月经周期
        if d1 == '' and d2 != '' and zq != '':
            d1 = date_add_num(d2, int(zq))
        elif zq == '' and d1 != '' and d2 != '':
            zq = date_sub_date(d1, d2)

        if record[5] != '' and d1 != '' and zq != '':
            s = date_sub_date(record[5], d1)
            if int(s) > int(zq):
                record.append(1)
            else:
                record.append(0)
        else:
            record.append(2)

    return results



def process():
    # results: medical_no, 编号, 科室类别，病历类别, 病历记录, 入院日期, 现病史，体温>37.5℃, 反跳通，肌紧张, 是否停经

    results = record_split('入院记录')
    print('record split finished...')
    results = extract_date(results)
    print('date extract finished...')
    results = extract_now_dieasehistory(results)
    print('nodh extract finished...')
    print('record filtr starting...')
    results = choose_records(results, 'data/labeled_ind_1432.txt')
    print('record filtr finished...')
    # results = extract_temp(results)
    # print('temp extract finished...')
    # results = extract_ftt(results)
    # print('ftt  extract finished...')
    # results = extract_jjj(results)
    # print('jjj  extract finished...')
    # results = extract_tj(results)
    # print('sftj extract finished...')

    return sorted(results, key=lambda x: x[0] + x[5])


def findPatternLine(pattern):
    """
    在整理的病史数据中查找模式
    """
    for record in record_split():
        for ind, line in enumerate(record[4].split('\n')):
            if pattern in line:
                print(line)


def load():
    results = []
    with open(r"data/tmp/mr_1432.txt", "r") as f:
        for row in f.readlines():
            results.append(row.strip().split(','))
    return results


if __name__ == "__main__":
    results = process()
    with open(r"data/tmp/mr_1432.txt", "w") as f:
        for row in results:
            # medical_no, 入院日期, 现病史，体温>37.5℃, 反跳通，肌紧张, 是否停经
            # f.write("%s,%s,%s,%s,%s,%s,%s\n" % (row[0], row[5], row[6], row[7], row[8], row[9], row[10]))
            f.write("%s||%s||%s||%s\n%s\n\n" % (row[0], row[1], row[2], row[3], row[4].strip()[:-1]))
        print('%s lines write to file data/tmp/mr_1432.txt' % len(results))



#

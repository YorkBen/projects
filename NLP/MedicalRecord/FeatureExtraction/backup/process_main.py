import sys
# print(sys.path)

from process_medical_record import process as proc_recr
from process_medical_record import load as load_recr
from process_inspect import process as proc_insp
from process_inspect import load_dict as load_insp_dict
from utils import *

"""
入口文件
"""

#20,11207 20210824
def join_inspec(data_records):
    """
    病历记录拼接检查数据。
    病历数据：
        输入：data_records
        格式：'medicalrecordno', '入院日期', '现病史', '体温>37.5℃', '反跳痛', '肌紧张', '是否停经'
    检查数据：
        输入：dict_insp
        格式：'medicalrecordno': ['检查日期', 检查名称, 指标值]
    输出：
        格式：'medicalrecordno', '入院日期', '现病史', '体温>37.5℃', '反跳痛', '肌紧张', '白细胞计数>10^9个/L', '中性粒细胞比例升高>70%'
    """
    dict_insp = load_insp_dict()
    print('join inspect records....')
    print('%s record to process' % len(data_records))
    print('dict length: %s' % len(dict_insp))
    has_insp_num = 0
    for i, r1 in enumerate(data_records):
        p1, p2 = '2', '2'
        if r1[1] != '' and r1[0] in dict_insp:
            insp_records = dict_insp[r1[0]]
            # 补齐日期中的','
            r1[1] = fix_datestr(r1[1], insp_records[0][0], ',')

            # 根据日期过滤检验记录
            r11_int = int(r1[1])
            candidate_records = []
            for r2 in insp_records:
                prior = r11_int - int(r2[0])
                if prior >= 0 and prior < 4:
                    candidate_records.append((prior, r2[1], r2[2]))
            candidate_records = sorted(candidate_records, key=lambda x: x[0])

            # 选取数据
            # p1 白细胞计数>10^9个/L, p2 中性粒细胞比例升高>70%,
            for r3 in candidate_records:
                if r3[1] == '白细胞计数>10^9个/L' and p1 == '2':
                    p1 = r3[2]
                elif r3[1] == '中性粒细胞比例升高>70%' and p2 == '2':
                    p2 = r3[2]

            if p1 != 2 or p2 != 2:
                has_insp_num = has_insp_num + 1
        r1.append(p1)
        r1.append(p2)

        if i % 1000 == 0 and i > 0:
            print('%s lines processed' % i)

    print('has insp num: %s' % has_insp_num)
    return data_records


# def load_labeled():
#     results = {}
#     with open(r'data/medical_labeled.txt', 'r') as f:
#         for idx, line in enumerate(f.readlines()):
#             if idx > 0:
#                 arr = line.strip().split(',')
#                 results[arr[0].strip()] = []
#
#     return results
#
#
# def load_record_inspec():
#     data = []
#     with open(r'data/medical_record_merge_results.txt', 'r') as f:
#         for idx, line in enumerate(f.readlines()):
#             if idx > 0:
#                 data.append(line.strip().split(','))
#
#     return data
#
# def join_record_labeled(data, labeled_dict):
#     count_in_dict = 0
#     count_total = len(labeled_dict)
#     labeled_keys = []
#     for key in labeled_dict:
#         labeled_keys.append(key)
#
#     for line in data:
#         key = line[2].strip()
#         if key in labeled_dict:
#             count_in_dict = count_in_dict + 1
#             # arr = labeled_dict[key]
#             # arr.extend(line[3:])
#             # arr.append(line[0])
#             arr = [line[0], line[1]]
#             arr.extend(line[3:])
#             labeled_dict[key] = arr
#
#     print(count_in_dict, count_total)
#     write_record_labeled(labeled_dict)
#
#
# def write_record_labeled(labeled_dict):
#     with open(r'data/medical_record_merge_labeled.txt', 'w') as f:
#         # f.write('medicalrecordno,现病史,厌食,恶心呕吐,右下腹疼痛,转移性右下腹疼痛,体温>37.5℃,反跳痛,白细胞计数>10^9个/L,中性粒细胞比例升高>70%,IDC11,主要诊断\n')
#         for key in labeled_dict:
#             arr = labeled_dict[key]
#             f.write('%s,%s\n' % (key, ','.join(arr)))
#             if len(arr) == 0:
#                 print(key)
#             # 替换2 -> 0, 替换 -1 -> 0
#             # 替换 -1 -> 2
#             # for k in [0, 1, 2, 3, 6, 7, 8, 9]:
#                 # if arr[k] == "2" or arr[k] == "-1":
#                 #     arr[k] = "0"
#                 # if arr[k] == "-1":
#                 #     arr[k] = "2"
#
#             # if len(arr) > 6:
#             #     f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (arr[-1], key, arr[0], arr[1], arr[2], arr[3], arr[6], arr[7], arr[8], arr[9], arr[4], arr[5]))
#             # else:
#             #     f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % ('', key, arr[0], arr[1], arr[2], arr[3], '', '', '', '', arr[4], arr[5]))


# # data_record = proc_recr()
# # data_insp = proc_insp()

data_records = load_recr()
data_records = join_inspec(data_records)

data_records_columns = ['medicalrecordno', '入院日期', '现病史', '体温>37.5℃', '反跳痛', '肌紧张', '是否停经', '白细胞计数>10^9个/L', '中性粒细胞比例升高>70%']
write_columns(data_records, data_records_columns, r'data/medical_record_results.txt')



#

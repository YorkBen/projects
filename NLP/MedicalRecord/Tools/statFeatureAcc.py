import sys

sys.path.append('../Lib')

from MRRecordUtil import *

file_name = r'D:\projects\NLP\MedicalRecord\FeatureExtracRegex\data\腹痛\临床特征_机器_人工.xls'
sheet_data_jq = load_sheet_arr_dict(file_name, 'Sheet1')
sheet_data_rg = load_sheet_arr_dict(file_name, 'Sheet2')
# sheet_data_mc = load_sheet_arr_dict(file_name, '名称映射')[0]

strict = True

keys = sheet_data_jq[0].keys()
result = []
for key in keys:
    key_result = {}
    for line1, line2 in zip(sheet_data_rg, sheet_data_jq):
        lbl_rg, lbl_jq = str(line1[key]), str(line2[key])
        if lbl_rg not in key_result:
            key_result[lbl_rg] = [0, 0]
        if strict:
            correct = 1 if lbl_jq == lbl_rg else 0
        else:
            correct = 1 if lbl_jq == lbl_rg or (lbl_jq == '0' and lbl_rg == '2') or (lbl_rg == '0' and lbl_jq == '2') else 0
        key_result[lbl_rg] = [key_result[lbl_rg][0] + 1, key_result[lbl_rg][1] + correct]

    # r = sheet_data_mc[key]
    r = key
    total, total_correct = 0, 0
    for n in ['0', '1', '2']:
        if n in key_result:
            r = '%s	%d	%d' % (r, key_result[n][0], key_result[n][1])
            r = '%s	%.2f' % (r, key_result[n][1] / key_result[n][0]) if key_result[n][0] > 0 else '%s	0' % (r)
            total = total + key_result[n][0]
            total_correct = total_correct + key_result[n][1]
        else:
            r = r + '	0	0	0'
    r = '%s	%.2f' % (r, total_correct/total) if total > 0 else '%s	0' % (r)
    result.append(r)

with open('机器人工比对结果.txt', 'w') as f:
    for line in result:
        f.write('%s\n' % line)

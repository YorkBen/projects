import sys

sys.path.append('../Lib')

from MRRecordUtil import *

keys = load_keys('../data/腹痛/labeled_ind_人机.txt', with_head=False)
# keys = load_keys('labeled_ind_测试.txt', with_head=False)
keys = set(['%s_%s' % (e[0], e[1]) for e in keys])


names = ['人机']
for name in names:
    sheet_data = load_sheet_arr_dict('门诊外院实验室特征提取.xlsx', name)
    result = []
    for row in sheet_data:
        key = '%s_%s' % (row['医保编号'], row['入院日期'])
        if key in keys:
            result.append(row)

    write_sheet_arr_dict(result, '%s.xlsx' % name, name, debug=False)
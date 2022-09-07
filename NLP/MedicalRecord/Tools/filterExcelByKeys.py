import sys

sys.path.append('../Lib')

from MRRecordUtil import *

keys = load_keys('../data/腹痛/labeled_ind_测试.txt', with_head=False)
# keys = load_keys('labeled_ind_测试.txt', with_head=False)
keys = list(['%s_%s' % (e[0], e[1]) for e in keys])
# keys = set(['%s' % (e[0]) for e in keys])

names = ['Sheet1']
for name in names:
    sheet_data = load_sheet_arr_dict(r'D:\projects\NLP\MedicalRecord\Tools\1.xls', name)
    result = []
    for row in sheet_data:
        key = '%s_%s' % (row['医保编号'], row['入院时间'])
        print(key)
        # key = '%d' % row['医保编号']
        if key in keys:
            result.append(row)

    write_sheet_arr_dict(result, '%s.xlsx' % name, name, debug=False)

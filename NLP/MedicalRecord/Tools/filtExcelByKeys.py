from MRRecordUtil import *

keys = load_keys('labeled_ind_测试.txt', with_head=False)
# keys = load_keys('labeled_ind_测试.txt', with_head=False)
keys = set(['%s_%s' % (e[0], e[1]) for e in keys])


names = ['Sheet1']
for name in names:
    sheet_data = load_sheet_arr_dict('训练集正则特征.xlsx', name)
    result = []
    for row in sheet_data:
        key = '%s_%s' % (row['YBBH'], row['入院时间'])
        if key in keys:
            result.append(row)

    write_sheet_arr_dict(result, '%s.xlsx' % name, name, debug=False)

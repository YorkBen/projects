# pip install openpyxl
from openpyxl import load_workbook
from openpyxl import Workbook
import json
import re

from Lib.MRRecordUtil import *
from Lib.InspRule import InspRule


def get_max_num(json_data):
    """
    '日常病程', '实验室数据', '超声', '放射', '病理', '医嘱'
    """
    key_arr = ['日常病程', '实验室数据', '超声', '放射', '病理', '医嘱']
    max_len_arr = [0, 0, 0, 0, 0, 0]
    for item in json_data:
        for idx, key in enumerate(key_arr):
            if key in item and len(item[key]) > max_len_arr[idx]:
                max_len_arr[idx] = len(item[key])
    return max_len_arr


def get_json_value(item, key):
    if key not in item:
        return ''
    else:
        return str(item[key])


def write_sheet_header(sheet, num_cs, num_fs):
    sheet.cell(1, 1).value = '医保编号'
    cn = 2
    for key, num in zip(['超声', '放射'], [num_cs, num_fs]):
        for i in range(1, num + 1):
            sheet.cell(1, cn).value = '%s_%d' % (key, i)
            cn = cn + 3


def write_sheet_data(sheet, diease, json_data, num_cs, num_fs):
    cn_starts = [2, 2 + num_cs * 3]
    rn = 2
    for ind, record in enumerate(json_data):
        sheet.cell(rn, 1).value = get_json_value(record, '医保编号')

        for key, cn in zip(['超声', '放射'], cn_starts):
            if key in record:
                item_arr = [(item['日期'], item['数据']) for item in record[key]]
                item_arr = sorted(item_arr, key=lambda x: x[0])

                for _, item_data in item_arr:
                    txt, match_txt = '', ''
                    for item2 in item_data:
                        arr = item2.split(',')
                        txt = txt + arr[-2] + arr[-1]

                    if re.search(diease_regex[diease], txt):
                        match_txt = re.search(diease_regex[diease], txt).group(0)
                        sheet.cell(rn, cn).value = 1
                        if diease in diease_suspect_regex and re.search(diease_suspect_regex[diease], txt):
                            match_txt = re.search(diease_suspect_regex[diease], txt).group(0)
                            sheet.cell(rn, cn).value = 3
                        elif re.search(diease_regex[diease] + inner + regex_suspect, txt):
                            match_txt = re.search(diease_regex[diease] + inner + regex_suspect, txt).group(0)
                            sheet.cell(rn, cn).value = 3
                    else:
                        sheet.cell(rn, cn).value = 0
                    sheet.cell(rn, cn + 1).value = match_txt
                    sheet.cell(rn, cn + 2).value = txt
                    cn = cn + 3

        rn = rn + 1


def compare_sheet_data(sheet, diease, json_data):
    cn_starts = [2, 7]
    rn = 2
    same_ct, total_ct = 0, 0
    for ind, record in enumerate(json_data):
        for key, start in zip(['超声', '放射'], cn_starts):
            if key in record:
                item_arr = [(item['日期'], item['数据']) for item in record[key]]
                item_arr = sorted(item_arr, key=lambda x: x[0])

                for idx2, (_, item_data) in enumerate(item_arr):
                    txt, match_txt, val = '', '', None
                    for item2 in item_data:
                        arr = item2.split(',')
                        txt = txt + arr[-2] + arr[-1]

                    if re.search(diease_regex[diease], txt):
                        val = 1
                        if diease in diease_suspect_regex and re.search(diease_suspect_regex[diease], txt):
                            val = 3
                        elif re.search(diease_regex[diease] + inner + regex_suspect, txt):
                            val = 3
                    else:
                        val = 0

                    total_ct = total_ct + 1
                    if sheet.cell(rn, start + idx2).value is None and val is None:
                        same_ct = same_ct + 1
                    elif sheet.cell(rn, start + idx2).value is not None and val is not None:
                        if (str(sheet.cell(rn, start + idx2).value) == '2' or str(sheet.cell(rn, start + idx2).value == '0')) and \
                            str(val) == '0':
                            same_ct = same_ct + 1
                        elif str(sheet.cell(rn, start + idx2).value) == '4' and str(val) == '0':
                            same_ct = same_ct + 1
                        elif str(sheet.cell(rn, start + idx2).value) == str(val):
                            same_ct = same_ct + 1
                        else:
                            # print(str(sheet.cell(rn, start + idx2).value), str(val))
                            pass
                    elif sheet.cell(rn, start + idx2).value is None and str(val) == '0':
                        same_ct = same_ct + 1
                    else:
                        # print(str(sheet.cell(rn, start + idx2).value), str(val))
                        pass

        rn = rn + 1

    return same_ct, total_ct


if __name__ == '__main__':
    isp = InspRule()
    data = load_sheet_arr_dict(r'data\腹痛\影像学正则测试数据.xlsx', 'Sheet1')

    for row in data:
        result = isp.process(row['文本'])
        for key in row.keys():
            if key != '文本':
                row[key] = result[key]

    write_sheet_arr_dict(data, r'data\腹痛\影像学正则测试数据_结果.xlsx', 'Sheet1')

    # # 加载json数据
    # json_data = ''
    # with open(r'data/汇总结果_%s.json' % postfix) as f:
    #     json_data = json.load(f, strict=False)
    # mrnos = load_mrnos(r'data/labeled_ind_1380.txt', with_head=False, sperator='	')
    # json_data = filter_json_values(json_data, mrnos)
    #
    # # # 写excel #####################
    # # workbook = load_workbook(r"data/result.xlsx")
    # wb = Workbook()
    # sheets = []
    # for idx, diease in enumerate(diease_regex.keys()):
    #     sheets.append(wb.create_sheet(diease, idx))
    #
    # _, _, num_cs, num_fs, _, _ = get_max_num(json_data)
    # for diease, sheet in zip(list(diease_regex.keys()), sheets):
    #     write_sheet_header(sheet, num_cs, num_fs)
    #     write_sheet_data(sheet, diease, json_data, num_cs, num_fs)
    #
    # # # 保存文档
    # wb.save(r'data\r_dpi_%s.xlsx' % postfix)
    # wb.close()



    # # 统计结果
    # workbook = load_workbook(r"C:\Users\Administrator\Desktop\训练集病例影像学.xlsx")
    # sheets = []
    # for idx, diease in enumerate(diease_regex.keys()):
    #     print(diease)
    #     same_ct, total_ct = compare_sheet_data(workbook[diease], diease, json_data)
    #     print(same_ct, total_ct, same_ct / total_ct)

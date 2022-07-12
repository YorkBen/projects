# pip install openpyxl
from openpyxl import load_workbook
import json

if __name__ == '__main__':
    postfix = '1432'

    # 加载json数据
    json_data = ''
    with open(r'data/汇总结果_%s.json' % postfix) as f:
        json_data = json.load(f, strict=False)

    def get_max_num(json_data):
        max_len, max_len2 = 0, 0
        for item in json_data:
            if '日常病程' in item and len(item['日常病程']) > max_len:
                max_len = len(item['日常病程'])
            if '实验室数据' in item and len(item['实验室数据']) > max_len2:
                max_len2 = len(item['实验室数据'])
        return max_len, max_len2

    def get_json_value(item, key):
        if key not in item:
            return ''
        else:
            return str(item[key])

    columns = ['医保编号', '性别', '年龄', '入院时间', '主诉', '现病史', '既往史', '手术外伤史', '药物过敏史',
                    '个人史', '婚育史', '月经史', '家族史', '体格检查', '专科情况（体检）', '门诊及院外重要辅助检查',
                    '病史小结', '出院诊断', '首次病程']

    # 日常病程 最大33个
    # 实验室数据 最大23个
    # max_len, max_len2 = get_max_num(json_data)
    # max_len, max_len2 = 33, 23
    # for i in range(1, max_len + 1):
    #     columns.append('日常病程%s' % i)
    # for i in range(1, max_len2 + 1):
    #     columns.append('实验室数据%s' % i)
    # columns.append('出院记录')

    # 写excel
    workbook = load_workbook(r"data/result.xlsx")
    sheet = workbook["Sheet1"]
    for ind, c in enumerate(columns):
        sheet.cell(1, 1+ind).value = c

    for ind, item in enumerate(json_data):
        rn = ind + 2
        cn = 0
        for ind_c, elem in enumerate(columns):
            cn = cn + 1
            if elem == '医保编号':
                sheet.cell(rn, cn).value = get_json_value(item, elem)
            elif elem == '首次病程':
                sheet.cell(rn, cn).value = get_json_value(item, elem)
            elif elem == '出院诊断':
                sheet.cell(rn, cn).value = get_json_value(item['出院记录'], elem) if '出院记录' in item else ''
            else:
                sheet.cell(rn, cn).value = get_json_value(item['入院记录'], elem)

        # 按照日期先后排列日常病程和实验室数据
        str_arr = []
        if '日常病程' in item:
            for item2 in item['日常病程']:
                str_arr.append((item2['DATE'], '日常病程\n' + str(item2)))
        for key in ['实验室数据', '超声', '放射', '病理', '医嘱']:
            if key in item:
                for item3 in item[key]:
                    str_arr.append((item3['日期'], key + '\n' + str(item3)))
        str_arr = sorted(str_arr, key=lambda x: x[0])

        for item23 in str_arr:
            cn = cn + 1
            sheet.cell(rn, cn).value = item23[1]

        # 出院记录
        cn = cn + 1
        sheet.cell(rn, cn).value = '出院记录\n' + get_json_value(item, '出院记录')

    workbook.save(r"data/r_%s.xlsx" % postfix)

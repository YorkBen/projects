# pip install openpyxl
# from openpyxl import load_workbook
import xlwt
import json
import re
import os
import sys
import argparse

sys.path.append('../Lib')

from MRRecordUtil import *

def get_max_num(json_data):
    max_len, max_len2 = 0, 0
    for item in json_data:
        if '日常病程' in item and len(item['日常病程']) > max_len:
            max_len = len(item['日常病程'])
        if '实验室数据' in item and len(item['实验室数据']) > max_len2:
            max_len2 = len(item['实验室数据'])
    return max_len, max_len2

    # return json.dumps(cur_item, indent=1, separators=(',', ':'), ensure_ascii=False)

def check_cs_part(part_str):
    """
    根据超声检查部位，返回子列号
    """
    if '腹股沟' in part_str:
        return 0
    elif re.search('((肾)|(输尿)|(膀胱)|(泌尿))', part_str):
        return 1
    elif re.search('((心)|(室壁))', part_str):
        return 2
    elif re.search('(前列腺)', part_str):
        return 3
    elif re.search('[肝胆胰脾]', part_str):
        return 4
    elif re.search('(胸水)', part_str):
        return 5
    elif re.search('(腹水)', part_str):
        return 6
    elif re.search('(腹部包块)', part_str):
        return 7
    elif re.search('(腹部大血管)', part_str):
        return 8
    elif re.search('((阑尾)|(肠管))', part_str):
        return 9
    elif re.search('(腹[^水])', part_str):
        return 10
    elif re.search('((附件)|(阴道)|(妇科))', part_str):
        return 11
    else:
        return 12


def check_ct_parts(part_str):
    """
    根据CT部位字符串，返回子列号，可能有多个
    """
    sub_inds = []
    if '胸' in part_str:
        sub_inds.append(0)
    if '腹' in part_str:
        sub_inds.append(1)
    if '上腹' in part_str:
        sub_inds.append(2)
    if '下腹' in part_str:
        sub_inds.append(3)
    if re.search('((肾)|(输尿)|(膀胱)|(泌尿)|(CTU))', part_str):
        sub_inds.append(4)
    if '盆腔' in part_str:
        sub_inds.append(5)
    if '颅脑' in part_str:
        sub_inds.append(6)

    if len(sub_inds) == 0:
        sub_inds.append(7)

    return sub_inds


def check_mr_parts(part_str):
    """
    根据MR部位字符串，返回子列号，可能有多个
    """
    sub_inds = []
    if '肝' in part_str or '胆' in part_str or 'MRCP' in part_str.upper():
        sub_inds.append(0)
    if '胰' in part_str:
        sub_inds.append(1)
    if '脾' in part_str:
        sub_inds.append(2)
    if '上腹' in part_str:
        sub_inds.append(3)
    if '下腹' in part_str:
        sub_inds.append(4)
    if '颅脑' in part_str:
        sub_inds.append(5)
    if '盆腔' in part_str:
        sub_inds.append(6)

    if len(sub_inds) == 0:
        sub_inds.append(7)

    return sub_inds


def check_dr_parts(part_str):
    """
    根据DR部位字符串，返回子列号，可能有多个
    """
    sub_inds = []
    if '胸' in part_str:
        sub_inds.append(0)
    if '腹' in part_str:
        sub_inds.append(1)

    if len(sub_inds) == 0:
        sub_inds.append(2)

    return sub_inds


def check_xx_parts(part_str):
    """
    根据X线部位字符串，返回子列号，可能有多个
    """
    sub_inds = [0]

    return sub_inds


def generate_cs_data(item):
    """
    从病历中按照columns的超声类型，生成对应的数据
    """
    cs_resstr_arr = ['' for k in range(len(columns['超声']))]
    # 生成各个超声检查的值
    if '超声' in item:
        for item_cs in item['超声']:
            for item_cs_sj in item_cs['数据']:
                if item_cs_sj.replace(',', '').strip() == '':
                    continue
                arr = item_cs_sj.split(',')
                if len(arr) == 3:
                    arr.append('')

                # 部位字符串
                part_str = arr[1] if arr[1].strip() != '' else arr[2].split('：')[0]
                # 检查部位
                sub_cn = check_cs_part(part_str)

                str_cs = '检查部位：%s\n检查所见：%s\n检查结论：%s\n\n' % (arr[1], arr[2], arr[3])
                cs_resstr_arr[sub_cn] = str_cs if cs_resstr_arr[sub_cn] == '' else cs_resstr_arr[sub_cn]

    return cs_resstr_arr


def generate_fs_data(item):
    """
    从病历中按照columns['CT', 'MR', 'DR']生成对应的数据
    """
    ct_resstr_arr = ['' for k in range(len(columns['CT']))]
    mr_resstr_arr = ['' for k in range(len(columns['MR']))]
    dr_resstr_arr = ['' for k in range(len(columns['DR']))]
    if '放射' in item:
        for item_fs in item['放射']:
            for item_fs_sj in item_fs['数据']:
                arr = item_fs_sj.split(',')
                if len(arr) == 3:
                    arr.append('')
                type, part = arr[0].upper(), arr[1]
                str_fs = '检查部位：%s\n检查所见：%s\n检查结论：%s\n\n' % (arr[1], arr[2], arr[3])
                if 'CT' in type:
                    for sub_cn in check_ct_parts(part):
                        ct_resstr_arr[sub_cn] = str_fs if ct_resstr_arr[sub_cn] == '' else ct_resstr_arr[sub_cn]
                elif 'MR' in type:
                    for sub_cn in check_mr_parts(part):
                        mr_resstr_arr[sub_cn] = str_fs if mr_resstr_arr[sub_cn] == '' else mr_resstr_arr[sub_cn]
                elif 'DR' in type or 'CR' in type or 'X线' in type:
                    for sub_cn in check_dr_parts(part):
                        dr_resstr_arr[sub_cn] = str_fs if dr_resstr_arr[sub_cn] == '' else dr_resstr_arr[sub_cn]

    return ct_resstr_arr, mr_resstr_arr, dr_resstr_arr


def generate_bl_data(item):
    """
    生成病理数据
    """
    str_bl = ''
    if '病理' in item:
        for item_bl in item['病理']:
            for item_bl_sj in item_bl['数据']:
                arr = item_bl_sj.split(',')
                str_bl = str_bl + '检查所见：%s\n诊断意见：%s\n\n' % (arr[0], arr[1])

    return str_bl


def generate_lab_data(item):
    """
    生成实验室数据
    """
    lab_resstr_arr = ['' for k in range(len(columns['实验室']))]
    if '检验' in item:
        for item_jy in item['检验']:
            for item_jy_sj in item_jy['数据']:
                arr = item_jy_sj.split(',')
                if len(arr) < 2:
                    continue

                sample, key = arr[1], arr[2]
                # 尿白细胞, 尿红细胞
                if sample == '尿液':
                    key = '尿' + key

                if key == '超敏C-反应蛋白':
                    key = 'C反应蛋白'
                    item_jy_sj = item_jy_sj.replace('超敏C-反应蛋白', 'C反应蛋白')

                if key not in columns['实验室']:
                    continue
                else:
                    append_str = item_jy['日期'] + ',' + item_jy_sj
                    idx = columns['实验室'].index(key)
                    lab_resstr_arr[idx] = append_str if lab_resstr_arr[idx] == '' else lab_resstr_arr[idx] + '\n' + append_str

    return lab_resstr_arr

columns = {
    '医保编号': '医保编号',
    '入院记录': ['患者姓名', '性别', '年龄', '入院时间', '主诉', '现病史', '既往史', '手术外伤史', '输血史', '药物过敏史', \
                    '个人史', '婚育史', '月经史', '家族史', '体格检查', '专科情况（体检）', '病史小结'],
    '出院诊断': '出院诊断',
    '超声': ['超声_腹股沟', '超声_泌尿', '超声_心脏', '超声_前列腺', '超声_肝胆', '超声_胸水', '超声_腹水', '超声_腹部包块', '超声_腹部大血管', \
                '超声_阑尾及肠管探查', '超声_腹部', '超声_妇科', '超声_其它'],
    'CT': ['CT_胸部', 'CT_全腹', 'CT_上腹', 'CT_下腹', 'CT_泌尿', 'CT_盆腔', 'CT_颅脑', 'CT_其它'],
    'MR': ['MR_肝胆&MRCP', 'MR_胰', 'MR_脾', 'MR_上腹', 'MR_下腹', 'MR_颅脑', 'MR_盆腔', 'MR_其它'],
    'DR': ['DR_胸部', 'DR_腹部', 'DR_其它'],
    '病理': '病理',
    '实验室': ['中性粒细胞%', '白细胞', 'C反应蛋白', '降钙素原', '脂肪酶', '淀粉酶', 'β-绒毛膜促性腺激素', \
                '总胆红素', '天冬氨酸氨基转移酶', '丙氨酸氨基转移酶', '碱性磷酸酶', 'γ-谷氨酰转移酶', '血沉', '红细胞', '尿白细胞', '尿红细胞']
     # ,'辅助检查_外院'
}

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='2409', help='postfix num')
    parser.add_argument('-t', type=str, default='腹痛', help='数据类型')
    args = parser.parse_args()

    postfix = args.p
    data_type = args.t
    if not os.path.exists('../data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    print("postfix: %s, data_type: %s" % (data_type, postfix))
    if not os.path.exists('../data/%s/labeled_ind_%s.txt' % (data_type, postfix)):
        print('mrnos file: ../data/%s/labeled_ind_%s.txt not exists!' % (data_type, postfix))
        exit()

    ## 加载json数据 ############
    keys, json_data = load_data(r'../data/%s/汇总结果_%s.json' % (data_type, postfix), '../data/%s/labeled_ind_%s.txt' % (data_type, postfix))

    ## 写excel ############
    # workbook = load_workbook(r"data/result.xlsx")
    # sheet = workbook["Sheet1"]
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Sheet1")

    ## 写表头 #############
    write_sheet_row(sheet, 0, columns)

    ## 写内容 #############
    for ind, item in enumerate(json_data):
        if item is None:
            continue
        ct_arr, mr_arr, dr_arr = generate_fs_data(item)
        row_data = {
            '医保编号': get_json_value(item, '医保编号'),
            '入院记录': [get_json_value(item, ['入院记录', '病史小结', '患者姓名'])] + [get_json_value(item, ['入院记录', col]) for col in columns['入院记录'][1:]],
            '出院诊断': get_json_value(item, ['出院记录', '出院诊断']),
            '超声': generate_cs_data(item),
            'CT': ct_arr,
            'MR': mr_arr,
            'DR': dr_arr,
            '病理': generate_bl_data(item),
            '实验室': generate_lab_data(item)
        }
        write_sheet_row(sheet, ind+1, row_data)


    workbook.save(r'data/%s/病历数据_%s.xlsx' % (data_type, postfix))

    print('saving file to data/%s/病历数据_%s.xlsx' % (data_type, postfix))

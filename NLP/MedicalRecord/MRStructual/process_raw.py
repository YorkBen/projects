import os
from openpyxl import Workbook
import argparse
import re
import sys

sys.path.append('../Lib')

from FileUtil import load_file, load_dict
from MRRecordUtil import process_mr, load_mrno

"""
处理原始病历数据
"""
def process_common(mr_nos, in_path, out_file_path, data_type='超声'):
    print('processing data type: %s' % data_type)
    # 处理单层目录或者单个文件
    input_files = []
    if os.path.isdir(in_path):
        input_files = [os.path.join(in_path, file_name) for file_name in os.listdir(in_path)]
    else:
        input_files = [in_path]
    print('input files: %s' % ','.join(input_files))


    results = []
    for file_path in input_files:
        with open(file_path) as f:
            for line in f.readlines()[1:]:
                elems = line.strip().split(',')
                if data_type == '放射':
                    mr_no, date = elems[0], elems[1][2:10]
                elif data_type == '病理':
                    mr_no = elems[0]
                    pos = elems[1].find('2')
                    date = elems[1][pos:pos+4] + '0101'
                elif data_type == '医嘱':
                    mr_no, date = elems[0], elems[1][:10].replace('-', '')
                else:
                    mr_no, date = elems[0], elems[1][:8]

                if mr_nos is None or mr_no in mr_nos:
                    results.append([mr_no, date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: (x[0], x[1]))

    with open(out_file_path, "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='all', help='postfix num')
    parser.add_argument('-t', type=str, default='恶心呕吐', help='data type')
    args = parser.parse_args()

    postfix = args.p
    type = args.t

    # mr_nos = load_mrno('data/%s/labeled_ind_%s.txt' % (type, postfix), with_head=False)
    mr_nos = None # 处理所有

    # 处理病历数据
    process_mr(file_path=r"data/%s/medical_record.csv" % type, with_head=False,
                type_regex_and_outpath=[
                    ('入.*院记录', r"data/%s/tmp/mr_ry_%s.txt" % (type, postfix)),
                    ('出院记录', r"data/%s/tmp/mr_cy_%s.txt" % (type, postfix)),
                    ('首次病程', r"data/%s/tmp/mr_sc_%s.txt" % (type, postfix)),
                    ('日常病程', r"data/%s/tmp/mr_rc_%s.txt" % (type, postfix))
                ], mr_nos=mr_nos)

    # process_common(mr_nos, r"data/%s/超声" % type, r"data/%s/tmp/chaoshen_%s.txt" % (type, postfix), data_type='超声')
    # process_common(mr_nos, r"data/%s/放射" % type, r"data/%s/tmp/fangshe_%s.txt" % (type, postfix), data_type='放射')
    # process_common(mr_nos, r"data/%s/检验" % type, r"data/%s/tmp/is_%s.txt" % (type, postfix), data_type='检验')
    # process_common(mr_nos, r"data/%s/医嘱" % type, r"data/%s/tmp/yizhu_%s.txt" % (type, postfix), data_type='医嘱')
    # process_bingli(mr_nos, r"data/%s/病理" % type, r"data/tmp/bingli_%s.txt" % postfix)

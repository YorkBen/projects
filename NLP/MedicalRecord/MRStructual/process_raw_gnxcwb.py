import os
from openpyxl import Workbook
import argparse
import re
import sys

sys.path.append('../Lib')

from FileUtil import load_file, load_dict
from MRRecordUtil import load_mrno

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
                    text = ','.join(elems[2:])
                elif data_type == '病理':
                    mr_no, date = elems[0], ''
                    text = ','.join(elems[1:])
                elif data_type == '医嘱':
                    mr_no, date = elems[0], elems[1][:10].replace('-', '')
                    text = ','.join(elems[2:])
                elif data_type == '其它检查':
                    mr_no, date = elems[0], ''
                    text = ','.join(elems[2:])
                else:
                    mr_no, date = elems[0], elems[1][:8]
                    text = ','.join(elems[2:])

                if mr_nos is None or mr_no in mr_nos:
                    results.append([mr_no, date, text])

    results = sorted(results, key=lambda x: (x[0], x[1]))

    with open(out_file_path, "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


"""
处理黄疸的病历
"""
def process_mr(file_path, out_path, with_head=True, mr_nos=None, num_fields=4):
    """
    处理病历数据，从中挑选出入院记录、出院记录、首次病程记录、日常病程记录等等
    """
    # ###debug####
    # match_dict = {mr_no:0 for mr_no in list(mr_nos)}
    # ###########
    print("处理病历数据...")
    medical_records = []
    mr_item, mr_cnt, skip = [], '', False
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx > 0 and idx % 10000 == 0:
                print('%s lines processed!' % (idx))

            # 第一行
            if idx == 0 and with_head:
                continue
            else:
                if ord(line[0]) == 65279: #BOM字符
                    line = line[1:]

            # 空行
            line = line.strip()
            if len(line) == 0:
                continue

            # 没有空格
            pos = 0
            for match_word in ['住院志病区', '出院记录病区', '病程记录病区', '病程记录姓名', '   ']:
                pos = line[:100].find(match_word)
                if pos != -1:
                    pos = pos + len(match_word) - 2
                    break

            part1 = line[:pos]
            part2 = line[pos:].strip()
            part2 = re.sub(r'(_nbsp_)+', ' ', part2)
            part2 = re.sub('[ ]{2,}', ' ', part2)
            part2 = re.sub(r';[; ]+;', '', part2)
            mr_item = part1.split(',')
            mr_item.append(mr_item[2])
            mr_item.append(part2)

            medical_records.append(mr_item)


    # 写数据
    with open(out_path, "w", encoding="utf-8") as f:
        for record in medical_records:
            f.write("%s\n%s\n\n" % ('||'.join(record[:-1]), record[-1]))
        print('%s lines write to file %s' % (len(medical_records), out_path))


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
    process_mr(file_path=r"data/%s/tmp/mr_ry.txt" % (type), out_path=r"data/%s/tmp/mr_ry_%s.txt" % (type, postfix), with_head=False)
    process_mr(file_path=r"data/%s/tmp/mr_rc.txt" % (type), out_path=r"data/%s/tmp/mr_rc_%s.txt" % (type, postfix), with_head=False)
    process_mr(file_path=r"data/%s/tmp/mr_sc.txt" % (type), out_path=r"data/%s/tmp/mr_sc_%s.txt" % (type, postfix), with_head=False)
    process_mr(file_path=r"data/%s/tmp/mr_cy.txt" % (type), out_path=r"data/%s/tmp/mr_cy_%s.txt" % (type, postfix), with_head=False)

    # process_common(mr_nos, r"data/%s/超声.csv" % type, r"data/%s/tmp/chaoshen_%s.txt" % (type, postfix), data_type='超声')
    # process_common(mr_nos, r"data/%s/放射.csv" % type, r"data/%s/tmp/fangshe_%s.txt" % (type, postfix), data_type='放射')
    # process_common(mr_nos, r"data/%s/检验.csv" % type, r"data/%s/tmp/is_%s.txt" % (type, postfix), data_type='检验')
    # process_common(mr_nos, r"data/%s/其它检查.csv" % type, r"data/%s/tmp/qitajiancha_%s.txt" % (type, postfix), data_type='其它检查')
    # process_common(mr_nos, r"data/%s/病理.csv" % type, r"data/%s/tmp/bingli_%s.txt" % (type, postfix))

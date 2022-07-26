import os
import re
import argparse
from Lib.MRRecordUtil import process_mr

"""
检查数据特征提取
"""

def load_mrno(file_path, with_head=True):
    """
    提取mrnos，文件的第一个字段是mrnos
    """
    mr_no = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head:
                continue
            mr_no.append(line.strip().split('	')[0])

    return set(mr_no)


def process_yizhu(mr_nos, outfile_path, folder="yizhu"):
    """
    提取医嘱数据
    """
    print("处理医嘱数据...")
    data_dir = r"data/%s/%s" % (data_type, folder)
    results = []
    for parent, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            if name.endswith('.csv'):
                with open(data_dir + "/" + name, encoding="utf-8") as f:
                    # print('processing file: %s' % name)
                    for line in f.readlines():
                        elems = line.strip().split(',')
                        if elems[0] in mr_nos:
                            date = elems[3][:10].replace('-', '') if elems[3] != '' else elems[2][:10].replace('-', '')
                            elems_ = [elems[0], date, ','.join(elems[1:])]
                            results.append(elems_)

    results = sorted(results, key=lambda x: x[0] + x[1])
    # nolist = [r[0] for r in results]
    # nolist_ = list(set(nolist))
    # print(len(nolist), len(nolist_))
    outfile_path = 'data/%s/tmp/%s' % (data_type, outfile_path)
    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))


def process_inspect(mr_nos, outfile_path, folder="inspect"):
    """
    处理检验数据
    """
    print("处理检验数据...")
    data_dir = r"data/%s/%s" % (data_type, folder)
    results = []
    for parent, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            if name.endswith('.csv'):
                with open(data_dir + "/" + name, encoding="utf-8") as f:
                    # print('processing file: %s' % name)
                    for line in f.readlines():
                        elems = line.strip().split(',')
                        if elems[0] in mr_nos:
                            elems[1] = elems[1][:8]
                            results.append(elems)

    results = sorted(results, key=lambda x: x[0] + x[1] + x[2] + x[3] + x[4])
    # nolist = [r[0] for r in results]
    # nolist_ = list(set(nolist))
    # print(len(nolist), len(nolist_))
    outfile_path = 'data/%s/tmp/%s' % (data_type, outfile_path)
    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))


def process_chaoshen(mr_nos, file_name, outfile_path):
    """
    处理超声数据
    """
    print("处理超声数据...")
    results = []
    file_path = r"data/%s/%s" % (data_type, file_name)
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][:8]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])
    outfile_path = 'data/%s/tmp/%s' % (data_type, outfile_path)
    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))


def process_fangshe(mr_nos, file_name, outfile_path):
    """
    处理放射数据
    """
    print("处理放射数据...")
    results = []
    file_path = r"data/%s/%s" % (data_type, file_name)
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][2:10]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])
    outfile_path = 'data/%s/tmp/%s' % (data_type, outfile_path)
    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))


def process_bingli(mr_nos, file_name, outfile_path):
    """
    处理病理数据
    """
    print("处理病理数据...")
    results = []
    file_path = r"data/%s/%s" % (data_type, file_name)
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos and elems[1] != '':
                pos = elems[1].find('2')
                if pos >= 0:
                    date = elems[1][pos:pos+4] + '0101'
                    results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])
    outfile_path = 'data/%s/tmp/%s' % (data_type, outfile_path)
    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))



if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='2409', help='postfix num')
    parser.add_argument('-t', type=str, default='腹痛', help='数据类型')
    args = parser.parse_args()

    postfix = args.p
    data_type = args.t
    if not os.path.exists('data/%s' % data_type):
        print('data type: %s not exists' % data_type)
        exit()
    print("postfix: %s, data_type: %s" % (data_type, postfix))
    if not os.path.exists('data/%s/labeled_ind_%s.txt' % (data_type, postfix)):
        print('mrnos file: data/%s/labeled_ind_%s.txt not exists!' % (data_type, postfix))
        exit()


    # 加载mrnos
    mr_nos = load_mrno('data/%s/labeled_ind_%s.txt' % (data_type, postfix), with_head=False)

    process_yizhu(mr_nos, folder="医嘱", outfile_path=r"yz_%s.txt" % postfix)
    process_inspect(mr_nos, folder="检验", outfile_path=r"is_%s.txt" % postfix)
    process_chaoshen(mr_nos, file_name=r"超声.csv", outfile_path=r"chaoshen_%s.txt" % postfix)
    process_fangshe(mr_nos, file_name=r"放射.csv", outfile_path=r"fangshe_%s.txt" % postfix)
    process_bingli(mr_nos, file_name=r"病理.csv", outfile_path=r"bingli_%s.txt" % postfix)
    process_mr(file_path=r"data/%s/medical_record.csv" % data_type, with_head=True,
                type_regex_and_outpath=[
                    ('入.*院记录', r"data/%s/tmp/mr_ry_%s.txt" % (data_type, postfix)),
                    ('出院记录', r"data/%s/tmp/mr_cy_%s.txt" % (data_type, postfix)),
                    ('首次病程', r"data/%s/tmp/mr_sc_%s.txt" % (data_type, postfix)),
                    ('日常病程', r"data/%s/tmp/mr_rc_%s.txt" % (data_type, postfix))
                ], mr_nos=mr_nos, num_fields=5)

#

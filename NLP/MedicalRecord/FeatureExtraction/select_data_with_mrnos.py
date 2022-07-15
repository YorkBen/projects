import os
import re
import argparse

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
            mr_no.append(line.split('	')[0])

    return set(mr_no)


def process_yizhu(mr_nos, outfile_path, folder="yizhu", file_prefix='ftyz'):
    """
    提取医嘱数据
    """
    print("处理医嘱数据...")
    data_dir = r"data/%s" % folder
    results = []
    for parent, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            if name.endswith('.csv') and name.startswith(file_prefix):
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

    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))


def process_inspect(mr_nos, outfile_path, folder="inspect", file_prefix='ftjy'):
    """
    处理检验数据
    """
    print("处理检验数据...")
    data_dir = r"data/%s" % folder
    results = []
    for parent, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            if name.endswith('.csv') and name.startswith(file_prefix):
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

    with open(outfile_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), outfile_path))


def process_chaoshen(mr_nos, file_path, out_file_path):
    """
    处理超声数据
    """
    print("处理超声数据...")
    results = []
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][:8]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(out_file_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


def process_fangshe(mr_nos, file_path, out_file_path):
    """
    处理放射数据
    """
    print("处理放射数据...")
    results = []
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][2:10]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(out_file_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


def process_bingli(mr_nos, file_path, out_file_path):
    """
    处理病理数据
    """
    print("处理病理数据...")
    results = []
    with open(file_path, encoding="utf-8") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                pos = elems[1].find('2')
                date = elems[1][pos:pos+4] + '0101'
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(out_file_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


def process_mr(mr_nos, file_path, with_head=True, type_regex_and_outpath=[('入.*院记录', r"data/tmp/mr.txt")]):
    """
    处理病历数据，从中挑选出入院记录、出院记录、首次病程记录、日常病程记录等等
    """
    # ###debug####
    # match_dict = {mr_no:0 for mr_no in list(mr_nos)}
    # ###########

    print("处理病历数据...")
    medical_records = [[] for i in range(len(type_regex_and_outpath))]
    mr_item, mr_cnt, skip = [], '', True
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head:
                continue

            elems = line.split(',')
            if len(elems) >= 4:
                # 有结果要写入
                if len(mr_item) > 0 and not skip:
                    for idx, item in enumerate(type_regex_and_outpath):
                        # ###debug####
                        # if mr_item[0] == '20051022':
                        #     print(item[0], mr_item[-1], re.search(item[0], mr_item[-1]))
                        #
                        # if item[0] == '出院记录':
                        #     match_dict[mr_item[0]] = 1
                        # ##########

                        if re.search(item[0], mr_item[-2]):
                            # if item[0] == '出院记录':
                            #     print(mr_item[-2])
                            medical_records[idx].append(mr_item)

                skip = False if elems[0] in mr_nos else True
                mr_item = elems
                mr_item[-1] = mr_item[-1].replace('\"', '')

                # ###debug####
                # if elems[0] == '20051022':
                #     print(skip, line)
                # ###########
            elif line.strip() != '':
                # 到达一个病历的结束行
                # 保留\n在后面作为行分割使用
                if line.strip().endswith('\"'):
                    mr_item[-1] = mr_item[-1] + line[:-1]
                else:
                    mr_item[-1] = mr_item[-1] + line

    # 写数据
    for idx, records in enumerate(medical_records):
        with open(type_regex_and_outpath[idx][1], "w", encoding="utf-8") as f:
            for row in records:
                str = '||'.join(row[:-1])
                f.write("%s\n%s\n\n" % (str, row[-1].strip()[:-1]))
        print('%s lines write to file %s' % (len(records), type_regex_and_outpath[idx][1]))

    # ###debug####
    # for key in match_dict.keys():
    #     if match_dict[key] == 0:
    #         print(key)
    # ############



if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-p', type=str, default='2409', help='postfix num')
    args = parser.parse_args()

    postfix = args.p
    print("postfix: %s" % (postfix))
    if not os.path.exists('data/labeled_ind_%s.txt' % postfix):
        print('mrnos file: data/labeled_ind_%s.txt not exists!' % postfix)
        exit()

    # 加载mrnos
    mr_nos = load_mrno('data/labeled_ind_%s.txt' % postfix)

    process_yizhu(mr_nos, folder="yizhu", file_prefix='ftyz', outfile_path=r"data/tmp/yz_%s.txt" % postfix)
    process_inspect(mr_nos, folder="inspect", file_prefix='ftjy', outfile_path=r"data/tmp/is_%s.txt" % postfix)
    process_chaoshen(mr_nos, file_path=r"data/超声.csv", out_file_path=r"data/tmp/chaoshen_%s.txt" % postfix)
    process_fangshe(mr_nos, file_path=r"data/放射.csv", out_file_path=r"data/tmp/fangshe_%s.txt" % postfix)
    process_bingli(mr_nos, file_path=r"data/病理.csv", out_file_path=r"data/tmp/bingli_%s.txt" % postfix)
    process_mr(mr_nos, file_path=r"data/medical_record.csv", with_head=True,
                type_regex_and_outpath=[
                    ('入.*院记录', r"data/tmp/mr_ry_%s.txt" % postfix),
                    ('出院记录', r"data/tmp/mr_cy_%s.txt" % postfix),
                    ('首次病程', r"data/tmp/mr_sc_%s.txt" % postfix),
                    ('日常病程', r"data/tmp/mr_rc_%s.txt" % postfix)
                ])

#

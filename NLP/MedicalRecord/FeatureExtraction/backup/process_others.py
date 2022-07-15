import os
import re

"""
检查数据特征提取
"""

def is_float(str):
    pattern = re.compile(r'^[-+]?[0-9]\d*\.\d*$')
    if pattern.match(str):
        return True
    else:
        return False


def load_mrno(file_path):
    mr_no = []
    with open(file_path) as f:
        for line in f.readlines():
            mr_no.append(line.split('	')[0])

    return set(mr_no)


def process_chaoshen(mr_nos, file_path, out_file_path):
    results = []
    with open(file_path) as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][:8]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(out_file_path, "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


def process_fangshe(mr_nos, file_path, out_file_path):
    results = []
    with open(file_path) as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][2:10]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(out_file_path, "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


def process_bingli(mr_nos, file_path, out_file_path):
    results = []
    with open(file_path) as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                pos = elems[1].find('2')
                date = elems[1][pos:pos+4] + '0101'
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(out_file_path, "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file %s' % (len(results), out_file_path))


if __name__ == "__main__":
    # postfix = '1432'
    postfix = '2409'

    mr_nos = load_mrno('data/labeled_ind_%s.txt' % postfix)
    process_chaoshen(mr_nos, r"data/超声.csv", r"data/tmp/chaoshen_%s.txt" % postfix)
    process_fangshe(mr_nos, r"data/放射.csv", r"data/tmp/fangshe_%s.txt" % postfix)
    process_bingli(mr_nos, r"data/病理.csv", r"data/tmp/bingli_%s.txt" % postfix)

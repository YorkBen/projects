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


def load_mrno():
    mr_no = []
    with open('data/labeled_ind.txt') as f:
        for line in f.readlines():
            mr_no.append(line.split('	')[0])

    return set(mr_no)


def process_chaoshen(mr_nos):
    results = []
    with open(r"data/超声.csv") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][:8]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(r"data/tmp/chaoshen.txt", "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file data/tmp/chaoshen.txt' % len(results))


def process_fangshe(mr_nos):
    results = []
    with open(r"data/放射.csv") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                date = elems[1][2:10]
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(r"data/tmp/fangshe.txt", "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file data/tmp/fangshe.txt' % len(results))


def process_bingli(mr_nos):
    results = []
    with open(r"data/病理.csv") as f:
        for line in f.readlines():
            elems = line.strip().split(',')
            if elems[0] in mr_nos:
                pos = elems[1].find('2')
                date = elems[1][pos:pos+4] + '0101'
                results.append([elems[0], date, ','.join(elems[2:])])

    results = sorted(results, key=lambda x: x[0] + x[1])

    with open(r"data/tmp/bingli.txt", "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file data/tmp/bingli.txt' % len(results))


def trans_arr_tomap(data_inspect):
    result = {}
    for r in data_inspect:
        m_no = r[0]
        if m_no not in result:
            result[m_no] = []
        result[m_no].append(r[1:])

    return result


def load_txt(type):
    filename = ""
    if type == "超声":
        filename = "chaoshen.txt"
    elif type == "放射":
        filename = "fangshe.txt"
    elif type == "病理":
        filename = "bingli.txt"

    results = []
    with open(r"data/tmp/%s" % filename, "r") as f:
        for row in f.readlines():
            results.append(row.strip().split(','))
    return results


def load_dict(type):
    return trans_arr_tomap(load_txt(type))


if __name__ == "__main__":
    mr_nos = load_mrno()
    process_chaoshen(mr_nos)
    process_fangshe(mr_nos)
    process_bingli(mr_nos)

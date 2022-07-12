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

def process(mr_nos):
    data_dir = r"data/yizhu"
    results = []
    for parent, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            if name.endswith('.csv') and name.startswith('ftyz'):
                with open(data_dir + "/" + name) as f:
                    print('processing file: %s' % name)
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

    return results


if __name__ == "__main__":
    postfix = '1432'

    mr_nos = load_mrno('data/labeled_ind_%s.txt' % postfix)
    results = process(mr_nos)
    with open(r"data/tmp/yz_all_%s.txt" % postfix, "w") as f:
        for row in results:
            f.write("%s\n" % ','.join(row))
        print('%s lines write to file data/tmp/yz_all_%s.txt' % (len(results), postfix))

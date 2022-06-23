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

def process():
    data_dir = r"data/inspect"
    results = []
    for parent, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            if name.endswith('.csv') and name.startswith('ftjy'):
                with open(data_dir + "/" + name) as f:
                    print('processing file: %s' % name)
                    for line in f.readlines():
                        elems = line.strip().split(',')
                        if elems[4] in ['中性粒细胞%', '白细胞'] and elems[3] in ['', '末梢血', '血清', '全血']:
                            if '尿' in elems[2] or '便' in elems[2]:
                                continue

                            value = elems[5].strip()
                            value = float(value) if is_float(value) else None

                            if elems[4] == '中性粒细胞%':
                                if value is None:
                                    results.append((elems[0], elems[1][:8], '中性粒细胞比例升高>70%', 2))
                                elif value > 70:
                                    results.append((elems[0], elems[1][:8], '中性粒细胞比例升高>70%', 1))
                                else:
                                    results.append((elems[0], elems[1][:8], '中性粒细胞比例升高>70%', 0))
                            elif elems[4] == '白细胞':
                                if value is None:
                                    results.append((elems[0], elems[1][:8], '白细胞计数>10^9个/L', 2))
                                elif value > 10:
                                    results.append((elems[0], elems[1][:8], '白细胞计数>10^9个/L', 1))
                                else:
                                    results.append((elems[0], elems[1][:8], '白细胞计数>10^9个/L', 0))

    results = list(set(results))
    results = sorted(results, key=lambda x: x[0] + x[1])
    # nolist = [r[0] for r in results]
    # nolist_ = list(set(nolist))
    # print(len(nolist), len(nolist_))

    return results


def trans_inspect_tomap(data_inspect):
    result = {}
    for ind, r1 in enumerate(data_inspect):
        m_no = r1[0]
        if m_no not in result:
            result[m_no] = []
        l = result[m_no]
        l.append((r1[1], r1[2], r1[3]))
        result[m_no] = l

    # 对每个m_no排序
    for m_no in result:
        l = result[m_no]
        result[m_no] = sorted(l, key=lambda x: x[0])

    return result


def load():
    results = []
    with open(r"data/tmp/is_s1.txt", "r") as f:
        for row in f.readlines():
            results.append(row.strip().split(','))
    return results

def load_dict():
    return trans_inspect_tomap(load())

if __name__ == "__main__":
    results = process()
    with open(r"data/tmp/is_s1.txt", "w") as f:
        for row in results:
            f.write("%s,%s,%s,%s\n" % (row[0], row[1], row[2], row[3]))
        print('%s lines write to file data/inspect/all_2.csv' % len(results))

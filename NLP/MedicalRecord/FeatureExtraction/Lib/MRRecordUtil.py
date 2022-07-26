import re

def load_mrno(file_path, with_head=True, separator='	'):
    """
    提取mrnos，文件的第一个字段是mrnos
    """
    mr_no = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head:
                continue
            mr_no.append(line.strip().split(separator)[0])

    return set(mr_no)


def load_keys(file_path, with_head=True, separator='	'):
    """
    提取mrnos和入院日期，文件的第一个字段是mrnos
    """
    results = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head:
                continue
            arr = line.strip().split(separator)
            results.append((arr[0], arr[1]))

    return results


def process_mr(file_path, with_head=True, type_regex_and_outpath=[('出.*院记录', r"data/tmp/mr.txt")], mr_nos=None, num_fields=4):
    """
    处理病历数据，从中挑选出入院记录、出院记录、首次病程记录、日常病程记录等等
    """
    # ###debug####
    # match_dict = {mr_no:0 for mr_no in list(mr_nos)}
    # ###########
    print("处理病历数据...")
    medical_records = [[] for i in range(len(type_regex_and_outpath))]
    mr_item, mr_cnt, skip = [], '', False
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx > 0 and idx % 10000 == 0:
                print('%s lines processed!' % (idx))

            # 第一行
            if idx == 0 and with_head:
                continue

            # 空行
            line = line.strip()
            if line == '':
                continue

            # 是否是记录起始行
            head_line, line_items = False, []
            if re.match('[IP0-9]{6}', line[:6]):
                line_items = line.split(',')
                if len(line_items) >= num_fields and re.search('((记录)|(证明书)|(同意书)|(病程))', line_items[num_fields-2]):
                    head_line = True

            # 有结果要写入
            if head_line:
                if len(mr_item) > 0 and not skip:
                    # 匹配多个记录类型，要输出多次
                    for idx, (type_regex, _) in enumerate(type_regex_and_outpath):
                        # ###debug####
                        # if mr_item[0] == '20051022':
                        #     print(item[0], mr_item[-1], re.search(item[0], mr_item[-1]))
                        #
                        # if item[0] == '出院记录':
                        #     match_dict[mr_item[0]] = 1
                        # ##########

                        # if re.search(type_regex, mr_item[-2]):
                        # if type_regex == '出院记录':
                        #     print(mr_item[-2])

                        if re.search(type_regex, mr_item[-2]):
                            medical_records[idx].append(mr_item)


                skip = False if mr_nos is None or line_items[0] in mr_nos else True
                mr_item = line_items[:num_fields]
                mr_item[-1] = ' '.join(line_items[num_fields-1:]).replace('\"', '')
                # ###debug####
                # if elems[0] == '20051022':
                #     print(skip, line)
                # ###########
            else:
                # 到达一个病历的结束行
                # 保留\n在后面作为行分割使用
                if line[-1] == '\"':
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

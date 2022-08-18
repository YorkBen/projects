import re
import json
import xlrd
import xlwt

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

    return mr_no


def load_keys(file_path, with_head=True, separator='	'):
    """
    提取mrnos和入院日期，文件的第一个字段是mrnos，第二个字段是入院时间
    """
    results = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0 and with_head or line.strip() == '':
                continue

            arr = line.strip().split(separator)
            mr_no, ry_date = arr[0], ''
            if len(arr) == 2:
                ry_date = arr[1]
                if ry_date != '' and not re.match('[0-9]{8}', ry_date):
                    print('%s:%s line illegal!' % (idx, line))
                    exit()

            results.append((mr_no, ry_date))

    return results


def sel_records(json_data, keys):
    """
    根据医保编号和入院日期过滤json数据
    keys：[(mrno, 入院日期)] 或者 [mrno] 或者None
    """
    results = [None for i in range(len(keys))]
    for record in json_data:
        if record['入院时间'] != '':
            key = '%s_%s' % (record['医保编号'], record['入院时间'])
        elif record['入院记录']['入院时间'] != '':
            key = '%s_%s' % (record['医保编号'], record['入院记录']['入院时间'])
        else:
            key = '%s' % record['医保编号']

        if key in keys:
            results[keys.index(key)] = record

    return results


def load_data(json_file, labeled_file, with_head=False):
    """
    加载json和mrnos，并根据mrnos过滤json
    """
    json_data = ''
    with open(json_file, encoding='utf-8') as f:
        json_data = json.load(f, strict=False)

    if labeled_file is not None:
        keys = load_keys(labeled_file, with_head=with_head, separator='	')
        keys = ['%s_%s' % (e[0], e[1]) for e in keys]
        json_data = sel_records(json_data, keys)
    else:
        keys = ['%s_%s' % (record["医保编号"], record["入院时间"]) for record in json_data]

    print('keys length: %d, json data length: %d' % (len(keys), len(json_data)))

    return keys, json_data


def load_sheet_dict(workbook_path, sheet_name):
    """
    加载数据表格。第一列作为每行的key，第一行的第二列至最后一列作为每列的key，生成行列嵌套字典：
    {key_row: {key_col2: val2, key_col3: val3...}}
    workbook_path: excel路径
    sheet_name：表单名
    val_type: str, int
    """
    # workbook = load_workbook()
    # sheet = workbook[sheet_name]

    workbook = xlrd.open_workbook(workbook_path)
    # sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    sheet = workbook.sheet_by_name(sheet_name)

    results, cols = {}, []
    # 写入列名
    rn = 0
    for j in range(0, sheet.ncols):
        if sheet.cell_value(rn, j) is None:
            break
        cols.append(sheet.cell_value(rn, j))

    # 写入行名，及数据
    for rn in range(1, sheet.nrows):
        if sheet.cell_value(rn, 0) is None:
            break

        mrno = sheet.cell_value(rn, 0)
        results[mrno] = {}
        for j in range(1, len(cols) + 1):
            val = sheet.cell_value(rn, j+1) if sheet.cell_value(rn, j+1) is not None else ''
            results[mrno][cols[j]] = val

    return results


def load_sheet_arr_dict(workbook_path, sheet_name):
    """
    加载数据表格。每行数据存入数组，第一行的第二列至最后一列作为每列的key，生成字典数组：
    [{key_col1: val1, key_col2: val2...}]
    workbook_path: excel路径
    sheet_name：表单名
    """
    # workbook = load_workbook(workbook_path)
    # sheet = workbook[sheet_name]
    workbook = xlrd.open_workbook(workbook_path)
    sheet = workbook.sheet_by_name(sheet_name)

    results, cols = [], []
    # 写入列名
    rn = 0
    for cn in range(0, sheet.ncols):
        if sheet.cell_value(rn, cn) is None:
            break
        cols.append(sheet.cell_value(rn, cn))

    # 写入行名，及数据
    for rn in range(1, sheet.nrows):
        if sheet.cell_value(rn, 0) is None:
            break

        row = {}
        for cn, col in enumerate(cols):
            row[col] = sheet.cell_value(rn, cn) if sheet.cell_value(rn, cn) is not None else ''
        results.append(row)

    return results



def write_sheet_arr_dict(data, workbook_path, sheet_name, debug=True):
    """
    写入数据表格。数据格式：生成字典数组：[{key_col1: val1, key_col2: val2...}]。
    字典的key作为表头，每行写一行数据。如果数据有几个字段，每个表头之间需要间隔相应的字段。字段依次写入相邻列中。
    workbook_path: excel路径
    sheet_name：表单名
    """
    wb = Workbook()
    sheet = wb.create_sheet(sheet_name, 0)

    cols = list(data[0].keys())

    # 写表头
    cn = 1
    for col in cols:
        sheet.cell(1, cn).value = col
        # 根据col的值类型，来添加列间距
        val = data[0][col]
        if debug and (isinstance(val, list) or isinstance(val, tuple)):
            for e in val:
                cn = cn + 1
        else:
            cn = cn + 1

    # 写数据
    for r_idx, row in enumerate(data):
        rn, cn = r_idx + 2, 1
        for col in cols:
            # 根据col的值类型，来添加列间距
            val = row[col]
            if debug and (isinstance(val, list) or isinstance(val, tuple)):
                for e in val:
                    sheet.cell(rn, cn).value = e
                    cn = cn + 1
            else:
                if isinstance(val, list) or isinstance(val, tuple):
                    sheet.cell(rn, cn).value = val[0]
                else:
                    sheet.cell(rn, cn).value = val
                cn = cn + 1

    wb.save(workbook_path)
    wb.close()


def write_sheet_dict(data, workbook_path, sheet_name):
    """
    """
    pass


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

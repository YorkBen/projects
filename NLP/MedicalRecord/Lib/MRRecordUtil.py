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
    # workbook_path = r"data/高发病率腹痛疾病特征标注2022.6.23.xlsx"
    # sheet_name = "前500个疾病特征标注"
    workbook = load_workbook(workbook_path)
    sheet = workbook[sheet_name]

    results, cols = {}, []
    # 写入列名
    for j in range(1, sheet.max_column + 1):
        if sheet.cell(1, j).value is None:
            break
        cols.append(sheet.cell(1, j).value)

    # 写入行名，及数据
    for i in range(2, sheet.max_row + 1):
        if sheet.cell(i, 1).value is None:
            break

        mrno = sheet.cell(i, 1).value
        results[mrno] = {}
        for j in range(1, len(cols) + 1):
            val = sheet.cell(i, j+1).value if sheet.cell(i, j+1).value is not None else ''
            results[mrno][cols[j]] = val

    return results


def load_sheet_arr_dict(workbook_path, sheet_name):
    """
    加载数据表格。每行数据存入数组，第一行的第二列至最后一列作为每列的key，生成字典数组：
    [{key_col1: val1, key_col2: val2...}]
    workbook_path: excel路径
    sheet_name：表单名
    """
    # workbook_path = r"data/高发病率腹痛疾病特征标注2022.6.23.xlsx"
    # sheet_name = "前500个疾病特征标注"
    workbook = load_workbook(workbook_path)
    sheet = workbook[sheet_name]

    results, cols = [], []
    # 写入列名
    for j in range(1, sheet.max_column + 1):
        if sheet.cell(1, j).value is None:
            break
        cols.append(sheet.cell(1, j).value)

    # 写入行名，及数据
    for i in range(2, sheet.max_row + 1):
        if sheet.cell(i, 1).value is None:
            break

        row = {}
        for idx, col in enumerate(cols):
            row[col] = sheet.cell(i, idx+1).value if sheet.cell(i, idx+1).value is not None else ''
        results.append(row)

    return results


def write_sheet_row(sheet, rn, row_data):
    """
    数据按照字典的形式组织，写入。
    """
    cn = 0
    for key, col in row_data.items():
        if isinstance(col, list) or isinstance(col, tuple):
            for elem in col:
                # sheet.cell(1, ind).value = col2
                sheet.write(rn, cn, elem.strip())
                cn = cn + 1
        else:
            # sheet.cell(1, ind).value = col
            sheet.write(rn, cn, col.strip())
            cn = cn + 1


def generate_columns_by_dict(data_dict, expand=True):
    """
    使用数据字典，生成对应的列。
    expand: True，扩展列名，如果字典值是字符串，则列名为key；如果字典值是数组，则列名为等长key数组
            False，列名均为key
    """
    columns = {}
    for key, val in data_dict.items():
        if expand and (isinstance(val, list) or isinstance(val, tuple)):
            columns[key] = []
            for e in val:
                columns[key].append(key)
        else:
            columns[key] = key

    return columns


def write_sheet_arr_dict(data, workbook_path, sheet_name, debug=True):
    """
    写入数据表格。数据格式：生成字典数组：[{key_col1: val1, key_col2: val2...}]。
    字典的key作为表头，每行写一行数据。如果数据有几个字段，每个表头之间需要间隔相应的字段。字段依次写入相邻列中。
    workbook_path: excel路径
    sheet_name：表单名
    """
    print('Debug Mode Opened!' if debug else 'Debug Mode Closed!')
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)

    cols = list(data[0].keys())

    # 写表头
    write_sheet_row(sheet, 0, generate_columns_by_dict(data[0], expand=debug))

    # 写数据
    for rn, row_data in enumerate(data):
        if not debug:
            row_data_ = {}
            for key, val in row_data:
                row_data_[key] = val[0] if (isinstance(val, list) or isinstance(val, tuple)) else val
            row_data = row_data_

        write_sheet_row(sheet, rn+1, row_data)

    workbook.save(workbook_path)
    print('save file: %s' % workbook_path)


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

def write_lines(data, file_path):
    """
    将行写到文件
    """
    with open(file_path, "w") as f:
        for r in data:
            f.write(r + "\n")
    print('%s lines write to %s' % (len(data), file_path))


def write_columns(data, columns, file_path):
    """
    将表格数据加上列名写到文件
    """
    data.insert(0, columns)
    with open(file_path, 'w') as f:
        for line in data:
            line_str = [str(i) for i in line]
            f.write(','.join(line_str) + '\n')
    print('%s lines write to %s' % (len(data) - 1, file_path))


def load_file(file_path, with_head=True, separator=','):
    """
    加载文件
    """
    results = []
    with open(file_path) as f:
        for line in f.readlines():
            results.append(line.strip().split(separator))

    if with_head:
        return results[1:]
    else:
        return results


def load_grid(file_path, separator=','):
    """
    将表格加载到DataFrame中，第一行是表的列名
    """
    data = load_file(file_path, False, separator)
    columns = data[0]
    data_r = data[1:]
    df = pd.DataFrame(columns=columns, data=data_r)

    return df


def mergeRelabeled(relabeled_file, idx):
    """
    将人工修正过的标记数据合并到原始数据
    """
    e_dict = {}
    with open(relabeled_file, "r") as f:
        for line in f.readlines():
            arr = line.split(',')
            if len(arr) != 2:
                print('illegal line : %s' % line)
                continue
            e_dict[arr[0]] = arr[1].strip()

    results = []
    with open("data/medical_labeled2.txt", "r") as f:
        for line in f.readlines():
            arr = line.split(',')
            if len(arr) != 5:
                print('illegal line : %s' % line)
                continue
            if arr[0] in e_dict:
                arr[idx] = e_dict[arr[0]]
            results.append(','.join(arr).strip())

    write_lines(results, "data/medical_labeled3.txt")

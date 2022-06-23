import json
import re
import time
from utils import *

debug = False

def load_template(template_file):
    """
    加载正则匹配模板
    """
    template = None
    with open(template_file) as f:
        template = json.load(f, strict=False)

    return template

def load_records(data_file):
    """
    加载病史数据
    """
    records = []
    record = []
    with open(data_file) as f:
        for line in f.readlines():
            if line.strip() == '':
                if len(record) > 0:
                    records.append(record)
                    record = []
            else:
                record.append(line)
        records.append(record)

    print('load records: %s' % len(records))
    return records

# def split_record_by_keys(record, keys):
#     """
#     将一个record按照key拆分位不同的区段
#     """
#     result = {}
#     record_ind, record_len = 0, len(record)
#     for ind, key in enumerate(keys):
#         cnt = []
#         # 要么是下一个key，要么是不存在读到最后
#         next_key = keys[ind+1] if ind+1 < len(keys) else '##########????????'
#         # 找开始行
#         for i in range(record_ind, record_len):
#             if record[i].strip().startswith(key):
#                 cnt.append(record[i])
#                 record_ind = i + 1
#                 break
#
#         # 找结束行
#         for i in range(record_ind, record_len):
#             record_ind = i
#             if record[i].startswith(next_key):
#                 break
#             else:
#                 cnt.append(record[i])
#
#         # 附到结果
#         result[key] = cnt
#
#     return result

# def split_cnt_by_keys(str, keys):
#     """
#     将一整段病史文本按照keys拆分成段
#     """
#     result = {}
#     str_ind = 0
#     keys_len = len(keys)
#     for ind, key in enumerate(keys):
#         # 找开始行
#         start = str.find(key, str_ind)
#         # 通过两个后续关键词来定位
#         end = -1
#         if start != -1:
#             str_ind = start + len(key)
#             ind2 = ind + 1
#             while ind2 < keys_len:
#                 # 要么是下一个key，要么是不存在读到最后
#                 end = str.find(keys[ind2], str_ind)
#                 if end != -1:
#                     break
#                 ind2 = ind2 + 1
#             result[key] = str[start:end]
#         else:
#             result[key] = ''
#         if debug:
#             print(key, start, end)
#
#     return result


def find_key_in_str_prob(key, str):
    """
    寻找key在str中的可能位置和概率。
    key使用模糊匹配查找，最多匹配字符个数不少于原字符串的50%
    """
    result, key_len = [], len(key)
    pattern = '[%s]{%s,}' % (key, (key_len + 1) // 2)
    for a in re.finditer(pattern, str):
        span = a.span()
        result.append((span[0], (span[1] - span[0]) * 1.0 / key_len))
    result.append((-1, 0))
    return result


matrix_max_cache = {}   #i^j -> max_value
def calc_max(i, j, matrix, max_pos):
    """
    递归计算i行j列及子节点的最大概率
    """
    key = '%s^%s' % (i, j)
    if key not in matrix_max_cache:
        # 最后一行
        if i == len(matrix) - 1:
            matrix_max_cache[key] = (i, j, matrix[i][j][1])
        else:
            max_pos = matrix[i][j][0] if matrix[i][j][0] != -1 else max_pos
            max_arr = [(k, calc_max(i+1, k, matrix, max_pos)[2]) for k in range(0, len(matrix[i+1])) if matrix[i+1][k][0] > max_pos or matrix[i+1][k][0] == -1]
            if len(max_arr) == 0:
                matrix_max_cache[key] = (i, j, matrix[i][j][1])
            else:
                max_arr = sorted(max_arr, key=lambda x: x[1], reverse=True)
                max_val = matrix[i][j][1] + max_arr[0][1]
                matrix_max_cache[key] = (i+1, max_arr[0][0], max_val)

    # print(matrix_max_cache[key])
    return matrix_max_cache[key]


def find_path(matrix, keys):
    """
    动态规划后，找最大概率路径
    """
    max_pos = -1
    path = []
    i, j = 0, 0
    while i < len(matrix):
        max_pos = matrix[i][j][0] if matrix[i][j][0] != -1 else max_pos
        i, j, score = calc_max(i, j, matrix, max_pos)
        # 到达最后行
        if len(path) > 0 and len(path) == len(keys):
            break
        # 从第二行开始
        if i > 0:
            path.append((keys[i-1], matrix[i][j]))

    return path


def split_cnt_by_keys_dynamic(str, keys):
    """
    使用动态规划，计算最大概率方法来拆分。
    """
    matrix = [[(-1, 0)]]
    for ind, key in enumerate(keys):
        matrix.append(find_key_in_str_prob(key, str))

    if debug:
        for i in range(1, len(matrix)):
            print(i, '---->', keys[i-1], '---->', matrix[i])

    global matrix_max_cache
    matrix_max_cache = {}
    calc_max(0, 0, matrix, -1)
    path = find_path(matrix, keys)


    result = {}
    for i in range(len(path)):
        if i < len(path) - 1:
            result[path[i][0]] = str[path[i][1][0]:path[i+1][1][0]]
        else:
            result[path[i][0]] = str[path[i][1][0]:]

    return result


def find_key_value_pattern(str, key):
    """
    输入字符串和label值，返回value值。
    """
    # re.DOTALL 可以匹配换行符
    res = re.match('^.*?[%s]{%s,}[：:]?(.*)' % (key, (len(key)+1)//2), str, re.DOTALL)
    if res is not None:
        return res.group(1)
    else:
        return ''

# def find_yuejing_zhouqi_pattern(sentence):
#     """
#     提取月经周期
#     """
#     search_obj1 = re.search(r'月经.*周期([2-9][0-9])天', sentence)
#     if search_obj1 is not None:
#         # print(search_obj1[1], 1)
#         return search_obj1[1]
#
#     search_obj2 = re.search(r'月经规[律则].*/[0-9-]*([2-9][0-9])', sentence)
#     if search_obj2 is not None:
#         # print(search_obj2[1], 2)
#         return search_obj2[1]
#
#     search_obj3 = re.search(r'月经.*/[0-9-]*([2-9][0-9])', sentence)
#     if search_obj3 is not None:
#         # print(search_obj3[1], 3)
#         return search_obj3[1]
#
#     return ''
#
# def extract_tj(results):
#     """
#     提取停经特征
#     """
#     for record in results:
#         # record[6] 现病史, record[5] 入院日期
#         zq = find_yuejing_zhouqi_pattern(record[6])
#         d1 = findLabelDateValuePattern(record[6], '末次月经|lmp')
#         d2 = findLabelDateValuePattern(record[6], 'pmp')
#
#         # 根据pmp补全末次月经日期或者月经周期
#         if d1 == '' and d2 != '' and zq != '':
#             d1 = date_add_num(d2, int(zq))
#         elif zq == '' and d1 != '' and d2 != '':
#             zq = date_sub_date(d1, d2)
#
#         if record[5] != '' and d1 != '' and zq != '':
#             s = date_sub_date(record[5], d1)
#             if int(s) > int(zq):
#                 record.append(1)
#             else:
#                 record.append(0)
#         else:
#             record.append(2)
#
#     return results


# 当val中包含如下关键词时，说明是下一个内容，去掉
remove_keys = ['输血史：', '民族：', '婚育史：', '记录医师签名：', '上述病史记录已征得陈述者认同。', '医师：', '医师签名：']
def format_vals(val):
    # 去除病历中的[***]内容
    val = remove_squarebracket_cnt(val)
    # 去掉包含在移除键数组中的key
    val = remove_after_keys(val, remove_keys)
    # 去掉末尾的（
    val = re.sub(r'（\s*\n?$', '', val)
    # 去掉一个（
    val = re.sub(r'^\s*）\s*\n?$', '', val)

    return val


def process_by_keys(str, template):
    """
    递归函数，对所有的key->cnt，处理cnt的val识别
    str: 匹配文本
    template: 匹配模板
    """
    if debug:
        print(str)
    keys = list(template.keys())
    result = template.copy()
    # record_split = split_cnt_by_keys(str, keys)
    record_split = split_cnt_by_keys_dynamic(str, keys)

    # 拆分内容
    for key in keys:
        str = record_split[key]
        if isinstance(template[key], dict):
            result[key] = process_by_keys(str, template[key])
        else:
            val = find_key_value_pattern(str, key)
            # result[key] = val
            val = format_vals(val)
            result[key] = format_by_type(val, template[key]).strip()

    return result

def process(template_file, data_file):
    """
    主函数
    """
    template = load_template(template_file)
    records = load_records(data_file)
    mr_no, results = '', []
    for record in records:
        result = process_by_keys(''.join(record), template)
        for line in record:
            if '||' in line:
                mr_no = line.split('||')[0]
                break
        # print(json.dumps(result, indent=1, separators=(',', ':'), ensure_ascii=False))
        date_str = ''
        if '入院' in template_file:
            date_str = result['入院时间']
        elif '首次病程' in template_file:
            date_str = result['首次病程记录']
        elif '日常病程' in template_file:
            date_str = result['日常病程记录']
        elif '出院' in template_file:
            date_str = result['出院时间']

        print(result)
        return
        results.append((mr_no, result['入院时间'], result))

    return results

    #'output/%s.json' % data_file.split('/')[-1].replace('.txt', '')


def write_json(data, file_path):
    """
    将Json数据写入文件
    """
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=1, separators=(',', ':'), ensure_ascii=False))


if __name__ == '__main__':
    starttime = time.time()
    # process('data/template/入院.json', 'data/records/手术科室入院记录.txt')
    # process('data/template/入院.json', 'data/records/入院测试.txt')
    # process('data/template/入院.json', 'data/records/非手术科室入院记录.txt')
    # process('data/template/日常病程.json', 'data/records/日常病程记录.txt')
    # process('data/template/出院记录.json', 'data/records/出院记录.txt')
    process('data/template/首次病程.json', 'data/records/首次病程.txt')
    endtime = time.time()
    print('Process Using Time: %s Seconds!', endtime - starttime)

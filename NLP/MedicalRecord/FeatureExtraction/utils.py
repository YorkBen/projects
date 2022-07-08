import re
import datetime
import pandas as pd

def segmentSentenct(sentence, hint_re, seg_re=r'[，。；：,.?？!！]'):
    """
    切分句子，并找出有hint_word的短句
    """
    pattern = re.compile(seg_re)
    sentence = re.sub(pattern, '++++', sentence)
    segs = sentence.split('++++')
    length_delta = 6
    results = []
    for idx, seg in enumerate(segs):
        if re.search(hint_re, seg):
            str = seg
            if len(seg) < length_delta:
                idx2 = idx - 1
                while idx2 >= 0 and len(segs[idx2]) < length_delta:
                    str = str + segs[idx2]
                    idx2 = idx2 - 1
            results.append(str)

    return results


def check_negative(str):
    return re.search('|'.join(['无', '不', '未']), str) and not re.search('|'.join(['无.*诱因']), str)


def pre_post_match(str, hint_re, prefix_arr, postfix_arr, pre_post_pair):
    """
    对一个短句做匹配，前缀、后缀、前后缀同时存在三组关系满足一个即匹配。
    """
    split_pattern = re.compile(r'(.*)(' + hint_re + ')(.*)')
    match = re.search(split_pattern, str)
    pre, post = match.group(1), match.group(3)
    r = 0
    # 前缀
    if prefix_arr is not None:
        sobj = re.search('|'.join(prefix_arr), pre)
        if sobj and not check_negative(pre):
            r = 1

    # 后缀
    if postfix_arr is not None:
        for post_pattern in postfix_arr:
            if post_pattern in post and not check_negative(pre):
                r = 1
    # 前后缀对
    if pre_post_pair is not None:
        for pre_pattern, post_pattern in pre_post_pair:
            if pre_pattern in pre and post_pattern in post and not check_negative(pre):
                r = 1
    if prefix_arr is None and postfix_arr is None and pre_post_pair is None:
        if not check_negative(pre):
            r = 1

    return r, pre, post

def format_by_type(str, type):
    """
    根据type类型格式化字符串。
    type类型：
        1. num
        2. age
        3. sex
        4. date
        5. temprature
        6. 空
        7. regex正则表达式
    """
    if type == "":
        return str
    elif type == "num":
        return format_num(str)
    elif type == "age":
        return format_age(str)
    elif type == "sex":
        return format_sex(str)
    elif type == "date":
        return format_date(str)
    elif type == "temprature":
        return format_temprature(str)
    elif type != "":
        return format_regex(str, type)

def format_num(str, default=''):
    """
    字符串匹配数字
    """
    str = format_regex(str, re.compile(r'[0-9]*[一二三四五六七八九十]*'))
    str = str.replace('一', '1').replace('二', '2').replace('三', '3').replace('四', '4')
    str = str.replace('五', '5').replace('六', '6').replace('七', '7').replace('八', '8')
    str = str.replace('九', '9')

    if str == '':
        return defalut
    else:
        return str

def format_age(str):
    """
    字符串匹配年龄并返回数字
    """
    return format_regex(str, re.compile(r'1?[0-9]?[0-9]'))

def format_sex(str):
    return format_regex(str, re.compile(r'男|女'))

def format_date(str):
    """
    解析字符串中的日期，并做标准化输出
    """
    r = re.search(r'(20[0-9]{2})[年\./\-]([01]?[0-9])[月\./\-]?([0123]?[0-9])?', str, re.I)
    if r:
        return format_date_output(r[1], r[2], r[3])
    else:
        return ''

def format_temprature(str):
    """
    格式化温度
    """
    return format_regex(str, re.compile(r'[0-4]\d\.?\d℃?'))

def format_regex(str, pattern_str):
    match = re.search(pattern_str, str)
    if match is None:
        return ''
    else:
        return match.group(0)

def remove_squarebracket_cnt(str):
    """
    去除字符串中中括号及包含内容
    """
    return re.sub(r'\[.*?\]', "", str)

def remove_after_keys(str, keys):
    """
    字符串中，包含keys中任何一个的以及其后的字符串都去掉
    """
    for key in keys:
        pattern = re.compile(r'%s.*' % key, re.DOTALL)
        str = re.sub(pattern, "", str)
    return str

def findExistPattern(sentences, hint_re, prefix_arr=None, postfix_arr=None, pre_post_pair=None):
    """
    判断语句中是否有存在该症状
    """
    # 分段模式
    # seg_pattern = re.compile(r'[，。，；：,.^](.*?' + hint_word + '.*?)[，。；：.,]')
    # 前后缀切分模式
    results, segs, pres, posts = [], [], [], []
    for line in sentences:
        r, s, pre, post = 0, '', '', ''
        # for match in re.finditer(seg_pattern, line):
            # s = match.group(1)
        sent_segs = segmentSentenct(line, hint_re)
        if len(sent_segs) > 0:
            for k in range(len(sent_segs) - 1, -1, -1):
                s = sent_segs[k]
                r, pre, post = pre_post_match(s, hint_re, prefix_arr, postfix_arr, pre_post_pair)
                if r == 1:
                    break

        results.append(r)
        segs.append(s)
        pres.append(pre)
        posts.append(post)

    return results, segs, pres, posts


def findPosNegNonePattern(sentences, hint_re, prefix_arr=None, postfix_arr=None, pre_post_pair=None):
    """
    判断语句有肯定的症状条数、否定症状条数、没有提及的条数
    """
    results, segs, pres, posts = [], [], [], []
    for line in sentences:
        r, s, pre, post = 0, '', '', ''
        # for match in re.finditer(seg_pattern, line):
            # s = match.group(1)
        sent_segs = segmentSentenct(line, hint_re)
        if len(sent_segs) > 0:
            for k in range(len(sent_segs) - 1, -1, -1):
                s = sent_segs[k]
                r, pre, post = pre_post_match(s, hint_re, prefix_arr, postfix_arr, pre_post_pair)
                if r == 1:
                    break
        else:
            # 没有提及
            r = 2
        # if r != 2:
        #     print(s, r, pre, post)
        #     print(line)

        results.append(r)
        segs.append(s)
        pres.append(pre)
        posts.append(post)

    return results, segs, pres, posts


def format_date_output(d1, d2, d3):
    if d1 is None:
        d1 = '2021'
    result = d1
    result = result + '0' if len(d2) == 1 else result + ''
    result = result + d2
    if d3 is None:
        d3 = '15'
    result = result + '0' if len(d3) == 1 else result + ''
    result = result + d3

    return result


def date_sub_date(str1, str2, df1='%Y%m%d', df2='%Y%m%d'):
    """
    求两个日期str1 - str2间隔天数
    日期格式：yyyyMMdd
    """
    d1 = datetime.datetime.strptime(str1, df1)
    d2 = datetime.datetime.strptime(str2, df2)
    return (d1 - d2).days


def date_add_num(str1, num, df1='%Y%m%d', df2='%Y%m%d'):
    """
    求两个日期str1 + 间隔天数后的日期
    日期格式：yyyyMMdd
    """
    d1 = datetime.datetime.strptime(str1, df1)
    d2 = d1 + datetime.timedelta(days=num)
    return d2.strftime(df2)

def fix_datestr(str1, str2, tofixch):
    """
    用日期字符串str2修复日期字符串str1，两个日期字符串必须格式相同。
    tofixch为日期字符串中需要替换的字符，如','
    """
    if tofixch in str1:
        print(str1, str2)
        return str1.replace(tofixch, str2.index(tofixch))
    else:
        return str1

def findLabelDateValuePattern(sentence, label_re):
    """
    查找标签值模式
    """
    patterns = [
        re.compile(r'(20[0-9]{2})[年\./\-]([01]?[0-9])[月\./\-]([0123]?[0-9])'),
        re.compile(r'(20[0-9]{2})[年\./\-]([01]?[0-9])'),
        re.compile(r'([01]?[0-9])[月\./\-]([0123]?[0-9])')
    ]
    s_obj = re.search(label_re, sentence, re.I)
    if s_obj:
        pos = s_obj.span()[1]
        for idx, pattern in enumerate(patterns):
            for r in re.finditer(pattern, sentence):
                if r.span()[0] >= pos:
                    if idx == 0:
                        return format_date_output(r[1], r[2], r[3])
                    elif idx == 1:
                        return format_date_output(r[1], r[2], None)
                    elif idx == 2:
                        return format_date_output(None, r[1], r[2])

    return ''


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

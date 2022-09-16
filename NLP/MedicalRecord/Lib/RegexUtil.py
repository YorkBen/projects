import re
import datetime
import pandas as pd

class RegexUtil:
    def __init__(self):
        """
        """
        pass

    def format_by_type(self, str, type):
        """
        根据type类型格式化字符串。
        type类型：
            1. num
            2. age
            3. sex
            4. date
            5. temprature
            6. pressure
            7. 空
            8. regex正则表达式
        """
        if type == "":
            return str
        elif type == "num":
            return self.format_num(str)
        elif type == "age":
            return self.format_age(str)
        elif type == "sex":
            return self.format_sex(str)
        elif type == "date":
            return self.format_date(str)
        elif type == "temprature":
            return self.format_temprature(str)
        elif type == "pressure":
            return self.format_pressure(str)
        elif type != "":
            return self.format_regex(str, type)


    def replace_chinese_num(self, str):
        """
        将中文数字替换为数字
        """
        str = str.replace('一', '1').replace('二', '2').replace('三', '3').replace('四', '4')
        str = str.replace('五', '5').replace('六', '6').replace('七', '7').replace('八', '8')
        str = str.replace('九', '9')
        return str


    def format_num(self, str, default='', forl='first'):
        """
        字符串匹配数字
        forl：返回第一个还是最后一个
        """
        str = self.replace_chinese_num(str)
        arr = re.findall(r'[0-9]+[.]?[0-9]*', str)

        if len(arr) == 0:
            return default
        elif forl == 'first':
            return arr[0]
        elif forl == 'last':
            return arr[-1]


    def format_nums(self, str):
        """
        字符串找出所有数字
        """
        str = self.replace_chinese_num(str)
        return re.findall(r'[0-9]+[.]?[0-9]*', str)


    def format_age(self, str):
        """
        字符串匹配年龄并返回数字
        """
        return self.format_regex(str, re.compile(r'1?[0-9]?[0-9]'))

    def format_sex(self, str):
        return self.format_regex(str, re.compile(r'男|女'))

    def format_date(self, str):
        """
        解析字符串中的日期，并做标准化输出
        """
        r = re.search(r'(20[0-9]{2})[年\./\-]([01]?[0-9])[月\./\-]?(([0123]?[0-9])|[底初末中上下旬]{1,3})?', str, re.I)
        if r:
            return self.format_date_output(r[1], r[2], r[3])
        else:
            return ''

    def format_temprature(self, str):
        """
        格式化温度
        """
        return self.format_regex(str, re.compile(r'[0-4]\d\.?\d℃?'))

    def format_pressure(self, str):
        """
        格式化血压
        """
        return self.format_regex(str, re.compile(r'\d{2,3}/\d{2,3}[mmhgMMHG]*?'))

    def format_regex(self, str, pattern_str):
        match = re.search(pattern_str, str)
        if match is None:
            return ''
        else:
            return match.group(0)

    def remove_squarebracket_cnt(self, str):
        """
        去除字符串中中括号及包含内容
        """
        return re.sub(r'\[.*?\]', "", str)

    def remove_after_keys(self, str, keys):
        """
        字符串中，包含keys中任何一个的以及其后的字符串都去掉
        """
        for key in keys:
            pattern = re.compile(r'%s.*' % key, re.DOTALL)
            str = re.sub(pattern, "", str)
        return str


    def find_key_value_pattern(self, str, key, dotall=True):
        """
        输入字符串和label值，返回value值。
        """
        # re.DOTALL 可以匹配换行符
        if dotall:
            res = re.match('^.*?[%s]{%s,}[：:]?(.*)' % (key, (len(key)+1)//2), str, re.DOTALL)
        else:
            res = re.match('^.*?[%s]{%s,}[：:]?(.*)' % (key, (len(key)+1)//2), str)

        if res is not None:
            return res.group(1)
        else:
            return ''


    def segmentSentenct(self, sentence, hint_re, seg_re=r'[，。；：,.?？!！]'):
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


    def check_negative(self, str):
        return re.search('|'.join(['无', '不', '未']), str) and not re.search('|'.join(['无.*诱因']), str)


    def pre_post_match(self, str, hint_re, prefix_arr, postfix_arr, pre_post_pair):
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
            if sobj and not self.check_negative(pre):
                r = 1

        # 后缀
        if postfix_arr is not None:
            for post_pattern in postfix_arr:
                if post_pattern in post and not self.check_negative(pre):
                    r = 1
        # 前后缀对
        if pre_post_pair is not None:
            for pre_pattern, post_pattern in pre_post_pair:
                if pre_pattern in pre and post_pattern in post and not self.check_negative(pre):
                    r = 1
        if prefix_arr is None and postfix_arr is None and pre_post_pair is None:
            if not self.check_negative(pre):
                r = 1

        return r, pre, post


    def findExistPattern(self, sentences, hint_re, prefix_arr=None, postfix_arr=None, pre_post_pair=None):
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
            sent_segs = self.segmentSentenct(line, hint_re)
            if len(sent_segs) > 0:
                for k in range(len(sent_segs) - 1, -1, -1):
                    s = sent_segs[k]
                    r, pre, post = self.pre_post_match(s, hint_re, prefix_arr, postfix_arr, pre_post_pair)
                    if r == 1:
                        break

            results.append(r)
            segs.append(s)
            pres.append(pre)
            posts.append(post)

        return results, segs, pres, posts


    def findPosNegNonePattern(self, sentences, hint_re, prefix_arr=None, postfix_arr=None, pre_post_pair=None):
        """
        判断语句有肯定的症状条数、否定症状条数、没有提及的条数
        """
        results, segs, pres, posts = [], [], [], []
        for line in sentences:
            r, s, pre, post = 0, '', '', ''
            # for match in re.finditer(seg_pattern, line):
                # s = match.group(1)
            sent_segs = self.segmentSentenct(line, hint_re)
            if len(sent_segs) > 0:
                for k in range(len(sent_segs) - 1, -1, -1):
                    s = sent_segs[k]
                    r, pre, post = self.pre_post_match(s, hint_re, prefix_arr, postfix_arr, pre_post_pair)
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


    def format_date_output(self, d1, d2, d3):
        # d1
        now = str(datetime.datetime.now())
        if d1:
            result = d1
        else:
            if int(now[5:7]) > 6:
                result = now[:4]
            else:
                result = str(int(now[:4]) + 1)

        # d2
        result = result + '0' + d2 if len(d2) == 1 else result + d2

        # d3
        if d3 is not None:
            if '中上' in d3:
                d3 = '10'
            elif '中下' in d3:
                d3 = '20'
            elif '初' in d3 or '上旬' in d3:
                d3 = '05'
            elif '中' in d3:
                d3 = '15'
            elif '末' in d3 or '下旬' in d3:
                d3 = '25'
            else:
                d3 = self.format_num(d3)
                d3 = '15' if d3 == '' else d3
        else:
            d3 = '15'

        result = result + '0' + d3 if len(d3) == 1 else result + d3

        return result


    def date_sub_date(self, str1, str2, df1='%Y%m%d', df2='%Y%m%d'):
        """
        求两个日期str1 - str2间隔天数
        日期格式：yyyyMMdd
        """
        d1 = datetime.datetime.strptime(str1, df1)
        d2 = datetime.datetime.strptime(str2, df2)
        return (d1 - d2).days


    def date_add_num(self, str1, num, df1='%Y%m%d', df2='%Y%m%d'):
        """
        求两个日期str1 + 间隔天数后的日期
        日期格式：yyyyMMdd
        """
        d1 = datetime.datetime.strptime(str1, df1)
        d2 = d1 + datetime.timedelta(days=num)
        return d2.strftime(df2)

    def fix_datestr(self, str1, str2, tofixch):
        """
        用日期字符串str2修复日期字符串str1，两个日期字符串必须格式相同。
        tofixch为日期字符串中需要替换的字符，如','
        """
        if tofixch in str1:
            logging.debug('date fix', str1, str2)
            return str1.replace(tofixch, str2.index(tofixch))
        else:
            return str1

    def findLabelDateValuePattern(self, sentence, label_re):
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
                            return self.format_date_output(r[1], r[2], r[3])
                        elif idx == 1:
                            return self.format_date_output(r[1], r[2], None)
                        elif idx == 2:
                            return self.format_date_output(None, r[1], r[2])

        return ''


    def find_key_in_str_prob(self, key, str):
        """
        寻找key在str中的可能位置和概率。
        key使用模糊匹配查找，最多匹配字符个数不少于原字符串的50%
        """
        result, key_len = [], len(key)
        # 日期和普通key的pattern不一样
        pattern = '[%s]{%s,}' % (key, (key_len + 1) // 2) if key != "DATE" else r'(20[0-9]{2})[年\./\-]([01]?[0-9])[月\./\-]?([0123]?[0-9])?'
        for a in re.finditer(pattern, str):
            span = a.span()
            # DATE 的分数默认为1.0
            score = 1.0
            # 其余的分数为匹配长度 / 字符串长度
            if key != "DATE":
                score = (span[1] - span[0]) * 1.0 / key_len
            # 如果KEY前有/n换行符或者开始序号，则score增加
            if re.match(r'.*[1-9]+[、. ]+|.*\n', str[max(span[0] - 3, 0):span[0]]) is not None:
                score = score * 1.1

            result.append((span[0], score))

        # 最后加一个找不到的情况
        result.append((-1, 0))

        return result


    def print_2d_arr(self, arr):
        """
        打印二维数组
        """
        print_arr = []
        for elem in arr:
            tmp = [str(e) for e in elem]
            print_arr.append('[' + ','.join(tmp) + ']')
        return '[' + ','.join(print_arr) + ']'




#

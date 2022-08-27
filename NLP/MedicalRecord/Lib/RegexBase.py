"""
正则匹配基类
"""
import re

class RegexBase:
    def __init__(self):
        # inner_neg
        self.inner_neg = '[^，；、。无未不]{,4}'
        self.inner_neg2 = '[^，；。无未不]{,4}'
        self.inner_neg3 = '[^；。无未不]{,4}'
        self.inner_neg_x = '[^，；、。无未不]{,12}'
        self.inner_neg_x2 = '[^，；。无未不]{,12}'
        self.inner_neg_x3 = '[^；。无未不]{,12}'
        self.inner_neg_xx = '[^，；、。无未不]{,20}'
        self.inner_neg_xx2 = '[^，；。无未不]{,20}'
        self.inner_neg_xx3 = '[^；。无未不]{,20}'
        self.inner = '[^，；、。]*?'
        self.inner2 = '[^，；。]*?'
        self.inner3 = '[^；。]*?'
        self.regex_suspect_aft = '(？|[?]|(怀疑)|(待排)|(可能)|(风险))'
        self.regex_suspect_bfr = '(考虑)'


    def split_text(self, text, subcls_regex=None):
        # text = text.replace('，间断', ' 间断').replace('，持续', ' 持续').replace('，阵发', ' 阵发')
        # text = text.replace('，呈', ' 呈').replace('，为', ' 为')
        # text = text.replace('，', '。')
        if subcls_regex:
            results = re.split(subcls_regex, text)
        else:
            results = re.split('。|\n', text)

        if results[-1] == '':
            return results[:-1]
        else:
            return results


    def split_text_by_regex_dict(self, text, regex_dict, choose_types=None):
        """
        使用关键词划分句子，并以包含关键词和前后的标点符号来分割
        regex_dict: {'超声': 'regex', '放射': 'regex'}
        """
        # 找到匹配项
        word_poss = []
        for type, regex in regex_dict.items():
            for match in re.finditer(regex, text, re.I):
                if match:
                    word_poss.append((type, match.span()[0]))

        if len(word_poss) == 0:
            return []
        elif len(word_poss) == 1:
            if (choose_types is not None and word_poss[0][0] in choose_types) or choose_types is None:
                return [text]
            else:
                return []

        # 对匹配项排序
        word_poss = sorted(word_poss, key=lambda x: x[1])
        word_poss_merged = [[word_poss[0][0], word_poss[0][1], word_poss[0][1]]]
        for i in range(1, len(word_poss)):
            type, start = word_poss[i]
            type_l, _, _ = word_poss_merged[-1]
            if type != type_l:
                word_poss_merged.append([word_poss[i][0], word_poss[i][1], word_poss[i][1]])
            else:
                word_poss_merged[-1][2] = start

        # 拆分句子
        result = []
        sent_start, last_type, last_end = 0, word_poss_merged[0][0], word_poss_merged[0][2]
        for type, start, end in word_poss_merged[1:]:
            # print(text[last_end:start])
            pos1 = [e.span()[0] for e in re.finditer("[，。,]", text[last_end:start])]
            pos2 = [e.span()[0] for e in re.finditer('[；; ：:？ ]', text[last_end:start])]
            # pos1 = [e.span()[0] for e in re.finditer("((，)|(。)|(,))", text[last_start:start])]
            # pos2 = [e.span()[0] for e in re.finditer('((；)|(;)|(：)|(:)|(？))', text[last_start:start])]
            if len(pos1) > 0:
                pos = pos1[-1]
            elif len(pos2) > 0:
                pos = pos2[-1]
            else:
                continue
            # print(text[sent_start:(last_end+pos+1)])
            result.append((last_type, text[sent_start:(last_end+pos+1)]))
            sent_start, last_type, last_end = last_end+pos+1, type, end
        result.append((last_type, text[sent_start:]))

        # 过滤type
        if choose_types is not None:
            result = [text for type, text in result if type in choose_types]
        else:
            result = [text for type, text in result]

        return result


    def get_dict_value(self, dict, key, default=None):
        if key not in dict:
            return default
        else:
            return dict[key]


    def search_by_regex_simple(self, text, regex, subcls_regex=None):
        """
        简单通过正则匹配，不做复杂逻辑处理
        """
        if regex is not None:
            for t in self.split_text(text, subcls_regex):
                match1 = re.search(regex, t, re.I)
                if match1:
                    return True, match1

        return False, None


    def check_neg_word(self, text, word):
        """
        检查text中是否有否定词，word为主体词, text为文本片段。
        """
        pos_match = re.search(word, text)
        mt1_sp2 = pos_match.span()[1]
        match2 = re.search(r"(无[^痛])|(不[^详全均])|未|(否认)|(除外)|(排除)", text[:mt1_sp2])
        match3 = re.search(r"(不明显)|(阴性)|(排除)|(((未见)|无)(明显)?异常)|([(（][-—][)）])", text[mt1_sp2:])
        if match2 and not '诱因' in text[:mt1_sp2]:
            return True, match2
        elif match3:
            return True, match3
        else:
            return False, None


    def check_suspect_word(self, text, word):
        """
        检查text中是否有怀疑词，word为主体词, text为文本片段。
        """
        pos_match = re.search(word, text)
        mt1_sp2 = pos_match.span()[1]
        match2 = re.search(self.regex_suspect_bfr, text[:mt1_sp2])
        match4 = re.search(self.regex_suspect_aft, text[mt1_sp2:])
        if match2:
            return True, match2
        elif match4 and match4.span()[0] <= 2:
            return True, match4
        else:
            return False, None


    def search_by_regex(self, text, regex, negregex=None, suspregex=None, exclregex=None, default=2):
        # 正匹配正则
        pos_match, pneg_match, psusp_match, rt1 = None, None, None, default
        for t in self.split_text(text):
            if re.search(regex, t):
                regex2 = '[^,，；;！]*(' + regex + ')[^,，；;！]*'
                for match1 in re.finditer(regex2, t):
                    # 排除匹配
                    if exclregex:
                        b_exclude = False
                        for match_excl in re.finditer(exclregex, t):
                            if match_excl and ((match_excl.span()[0] >= match1.span()[0] and match_excl.span()[0] < match1.span()[1]) \
                                            or (match_excl.span()[1] > match1.span()[0] and match_excl.span()[1] <= match1.span()[1])):
                                b_exclude = True
                                break
                        if b_exclude:
                            continue

                    t_ = t[match1.span()[0]:match1.span()[1]]
                    pos_match, pneg_match, psusp_match, rt1 = None, None, None, default
                    # match1 = re.search(regex2, t)
                    pos_match = re.search(regex, t_)

                    # 否定
                    bNeg, bNeg_match = self.check_neg_word(t_, regex)
                    if bNeg:
                        pneg_match, rt1 = bNeg_match, 0
                    else:
                        rt1 = 1
                    #####被上面代码取代
                    # pos_match = re.search(regex, t_)
                    # mt1_sp2 = pos_match.span()[1]
                    # match2 = re.search(r"(无[^痛])|(不[^详全])|未|(否认)", t_)
                    # match3 = re.search(r"(不明显)|(阴性)|(排除)|(((未见)|无)(明显)?异常)|([(（][-—][)）])", t_)
                    # if match2 and match2.span()[1] < mt1_sp2 and not '诱因' in t_[:mt1_sp2]:
                    #
                    # elif match3 and match3.span()[0] >= mt1_sp2:
                    #     pneg_match, rt1 = match3, 0
                    # else:
                    #     rt1 = 1

                    # 待排、可能
                    bSusp, bSusp_match = self.check_suspect_word(t_, regex)
                    if bSusp:
                        psusp_match, rt1 = bSusp_match, 3

                    ##########被上面代码取代
                    # match4 = re.search(self.regex_suspect, t_)
                    # if match4 and match4.span()[0] >= mt1_sp2 and (match4.span()[0] - mt1_sp2) <= 2:
                    #     psusp_match, rt1 = match4, 3

                    if rt1 == 1:
                        break
        # print(text)
        # print(pos_match, pneg_match, psusp_match, rt1)

        # 否定匹配正则
        neg_match, rt2 = None, default
        if negregex:
            r_ss, match_ss = self.search_by_regex_simple(text, negregex)
            if r_ss:
                neg_match, rt2 = match_ss, 0

        # print(neg_match, rt2)

        # 怀疑匹配正则
        susp_match, rt3 = None, default
        if suspregex:
            r_ss, match_ss = self.search_by_regex_simple(text, suspregex)
            if r_ss:
                susp_match, rt3 = match_ss, 3

        # print(pos_match, pneg_match, psusp_match, rt1)
        # print(neg_match, rt2)
        # print(susp_match, rt3)

        ## 合并结果
        rt_type = 4
        if pos_match is not None and neg_match is not None:
            if pos_match.span()[1] - pos_match.span()[0] <= neg_match.span()[1] - neg_match.span()[0] - 4:
                rt_type = 1
            else:
                rt_type = 2
        elif pos_match is not None and susp_match is not None:
            if pos_match.span()[1] - pos_match.span()[0] <= susp_match.span()[1] - susp_match.span()[0] - 4:
                rt_type = 1
            else:
                rt_type = 3
        elif pos_match is not None:
            rt_type = 1
        elif neg_match is not None:
            rt_type = 2
        elif susp_match is not None:
            rt_type = 3


        if rt_type == 1:
            if pneg_match is not None:
                return rt1, pos_match, pneg_match
            else:
                return rt1, pos_match, psusp_match
        elif rt_type == 2:
            return rt2, neg_match, None
        elif rt_type == 3:
            return rt3, susp_match, None
        else:
            return default, None, None



if __name__ == '__main__':
    rb = RegexBase()
    text = '（包括日期、医疗机构、检查项目、结果）：2017-7-16我院血常规提示：白细胞 10.86 10^9个/L， 中性粒细胞% 93.60 % ，中性粒细胞计数10.17 10^9个/L ；2020-7-17 CT下腹部平扫提示：1.考虑阑尾炎可能，请结合临床病史及查体，建议必要时复查；2.右肾稍低密度灶，多考虑囊肿。'
    regex = '阑尾' + rb.inner_neg + '((炎)|(脓肿)|(穿孔))'
    rb.search_by_regex(text, regex)

"""
正则匹配基类
"""
import re

class RegexBase:
    def __init__(self):
        # inner_neg
        self.inner_neg = '[^，；、。无未不]{,4}'
        self.inner_neg_x = '[^，；、。无未不]{,8}'
        self.inner = '[^，；、。]*?'
        self.regex_suspect = '((？)|[?]|(怀疑)|(待排)|(可能))'


    def split_text(self, text):
        # text = text.replace('，间断', ' 间断').replace('，持续', ' 持续').replace('，阵发', ' 阵发')
        # text = text.replace('，呈', ' 呈').replace('，为', ' 为')
        # text = text.replace('，', '。')
        return text.split('。')


    def search_by_regex_simple(self, text, regex, rt=1, default=2):
        """
        简单通过正则匹配，不做复杂逻辑处理
        """
        rmatch, rt = None, default
        if regex is not None:
            for t in self.split_text(text):
                match1 = re.search(regex, t)
                if match1:
                    rmatch, rt = match1, rt
                    break

        return rmatch, rt


    def check_neg_word(self, text, word):
        """
        检查text中是否有否定词，word为主体词, text为文本片段。
        """
        pos_match = re.search(word, text)
        mt1_sp2 = pos_match.span()[1]
        match2 = re.search(r"(无[^痛])|(不[^详全])|未|(否认)", text)
        match3 = re.search(r"(不明显)|(阴性)|(排除)|(((未见)|无)(明显)?异常)|([(（][-—][)）])", text)
        if match2 and match2.span()[1] < mt1_sp2 and not '诱因' in text[:mt1_sp2]:
            return True, match2
        elif match3 and match3.span()[0] >= mt1_sp2:
            return True, match3
        else:
            return False, None


    def search_by_regex(self, text, regex, negregex=None, suspregex=None, default=2):
        # 正匹配正则
        pos_match, pneg_match, psusp_match, rt1 = None, None, None, default
        for t in self.split_text(text):
            if re.search(regex, t):
                regex2 = '[^,，；;！]*(' + regex + ')[^,，；;！]*'
                match1 = re.search(regex2, t)
                t_ = t[match1.span()[0]:match1.span()[1]]

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
                match4 = re.search(self.regex_suspect, t_)
                if match4 and match4.span()[0] >= mt1_sp2 and (match4.span()[0] - mt1_sp2) <= 2:
                    psusp_match, rt1 = match4, 3


        # 否定匹配正则
        neg_match, rt2 = self.search_by_regex_simple(text, negregex, rt=0, default=default)

        # 怀疑匹配正则
        susp_match, rt3 = self.search_by_regex_simple(text, suspregex, rt=3, default=default)


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

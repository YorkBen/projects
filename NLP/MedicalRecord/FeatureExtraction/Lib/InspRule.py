"""
影像学正则提取类
"""
import re
from Lib.RegexBase import RegexBase

class InspRule(RegexBase):
    def __init__(self):
        super(InspRule, self).__init__()

        # 疾病对应的正则
        self.diease_regex = {
            '急性阑尾炎': '阑尾' + self.inner_neg + '((炎)|(脓肿)|(穿孔))',
            '急性胰腺炎': '胰腺' + self.inner_neg + '((炎)|(脓肿)|(穿孔))',
            '肠梗阻': '肠梗阻',
            '异位妊娠': '宫外孕',
            '急性胆管炎': '胆管炎',
            '急性胆囊炎': '胆囊炎',
            '上尿路结石': '((肾)|(输尿管)|(膀胱))' + self.inner_neg + '结石',
            '消化性溃疡穿孔': '穿孔'
        }

        # 疾病待排对应的正则
        self.diease_suspect_regex = {
            '急性胆管炎': '胆管炎' + self.inner + '((？)|[?]|(怀疑)|(待排)|(可能)|(硬化性))'
        }


    def process(self, str):
        """
        处理单条字符串
        """
        # 结果
        result = {diease: (0, '', '') for diease in self.diease_regex.keys()}

        for diease in self.diease_regex.keys():
            rt, match1, match2 = self.search_by_regex(str, self.diease_regex[diease],
                            suspregex=self.diease_suspect_regex[diease] if diease in self.diease_suspect_regex else None)
            #
            # if re.search(self.diease_regex[diease], str):
            #     match_txt = re.search(self.diease_regex[diease], str).group(0)
            #     result[diease] = (1, match_txt, str)
            #     if diease in self.diease_suspect_regex and re.search(self.diease_suspect_regex[diease], str):
            #         match_txt = re.search(self.diease_suspect_regex[diease], str).group(0)
            #         result[diease] = (3, match_txt, str)
            #     elif re.search(self.diease_regex[diease] + self.inner + self.regex_suspect, str):
            #         match_txt = re.search(self.diease_regex[diease] + self.inner + self.regex_suspect, str).group(0)
            #         result[diease] = (3, match_txt, str)
            result[diease] = (rt, match1.group(0) if match1 is not None else '', match2.group(0) if match2 is not None else '', str)

        return result

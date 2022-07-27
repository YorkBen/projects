"""
实验室数据提取特征
"""
import re
from Lib.RegexBase import RegexBase
from Lib.Utils import Utils

class LabRule(RegexBase):
    def __init__(self):
        super(LabRule, self).__init__()

        self.utils = Utils()

        self.rules = {
            1: ('全血', '白细胞', 'gt', 9.5),
            2: ('全血', '白细胞', 'lt', 4),
            3: ('全血', '白细胞', 'gt', 10),
            4: ('全血', '中性粒细胞%', 'gt', 75),
            5: ('全血', 'C反应蛋白', 'gt', 10),
            6: ('全血', '血沉', 'gt', 26),
            7: ('全血', '降钙素原', 'gt', 0.1),
            8: ('血清', '降钙素原', 'gt', 0.1),
            9: ('血清', '脂肪酶', 'gt', 900),
            10: ('血清', '淀粉酶', 'gt', 405),
            11: ('血清', '天冬氨酸氨基转移酶', 'gt', 75),
            12: ('血清', '丙氨酸氨基转移酶', 'gt', 60),
            13: ('血清', '碱性磷酸酶', 'gt', 187.5),
            14: ('血清', 'γ-谷氨酰转移酶', 'gt', 90),
            15: ('血清', 'β-绒毛膜促性腺激素', 'gt', 10),
            16: ('血清', '总胆红素', 'gt', 22),
            17: ('尿液', '红细胞', 'gt', 28),
            18: ('尿液', '白细胞', 'gt', 17)
        }

        self.diease_rules = {
            '急性阑尾炎': [4, 1, 5, 7, 8],
            '急性胰腺炎': [9, 10],
            '异位妊娠': [15],
            '急性胆管炎': [2, 3, 5, 16, 11, 12, 13, 14],
            '急性胆囊炎': [1, 5, 6],
            '上尿路结石': [17],
            '急性肾盂肾炎': [18]
        }


    def get_diease_results(self, rule_results):
        """
        从规则结果来生成疾病结果。
        输入：
            {rule_id: 0,1,2}
        输出：
            {
                '急性阑尾炎': {
                    '白细胞': 0,1,2
                }
            }
        """
        diease_results = {}
        for diease in self.diease_rules.keys():
            diease_results[diease] = {}
            for rule_id in self.diease_rules[diease]:
                diease_results[diease][self.rules[rule_id][1]] = rule_results[rule_id]

        return diease_results


    def get_rule_results(self, rule_results):
        """
        从规则结果来生成规则名称结果。
        输入：
            {rule_id: 0,1,2}
        输出：
            {
                '全血_白细胞_大于_9.5': 0,1,2
            }
        """
        results = {}
        for rule_id in rule_results.keys():
            rule = self.rules[rule_id]
            rule = list(rule)
            rule[3] = str(rule[3])
            rule[2] = rule[2].replace('gt', '大于').replace('lt', '小于')
            rule_str = '_'.join(rule)
            results[rule_str] = rule_results[rule_id]

        return results


    def process_strc_items(self, items):
        """
        处理结构化实验室数据。
        输入：
            组套名称, 标本, 项目, 检验值, 单位, 参考值范围：血常规.hsCRP.SAA, 全血, C反应蛋白, 5.92, mg/L, 0-10
            输入为一个病历号对应的所有实验室数据。
        输出：
            所有特征提取结果
            {rule_id: 0,1,2}
        """
        # 按规则来计算
        rule_results = {k:2 for k in self.rules.keys()}
        for _, item_bb, item_name, item_val, item_unit, item_rg in items:
            for rule_id in self.rules.keys():
                rule_bb, rule_name, rule_type, rule_val = self.rules[rule_id]
                if rule_name == item_name and rule_bb == item_bb:
                    str_val = self.utils.format_num(item_val)
                    if str_val != '':
                        val = float(str_val)
                        if rule_type == 'gt' and val > rule_val or \
                            rule_type == 'lt' and val < rule_val:
                            rule_results[id] = 1
                        else:
                            if rule_results[id] != 1:
                                rule_results[id] = 0


        return rule_results


    def process_texts(self, texts):
        """
        从自由文本中提取实验室数据特征。
        输入：
            文本数组
        输出：
            所有特征提取结果
            {rule_id: 0,1,2}
        """






#

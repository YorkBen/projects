"""
实验室数据提取特征
"""
import re
from Lib.RegexBase import RegexBase
from Lib.Utils import Utils

class LabRule(RegexBase):
    def __init__(self, debug=False):
        super(LabRule, self).__init__()
        self.debug = debug
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

        self.rule_str_regex = {
            1: '(WBC[^；、，。]{,10})|(白细胞[^；、，。酶]{,10})',
            2: '(WBC[^；、，。]{,10})|(白细胞[^；、，。酶]{,10})',
            3: '(WBC[^；、，。]{,10})|(白细胞[^；、，。酶]{,10})',
            4: '(N[^a-df-tv-z]([^，,、；]{,10})%)|((中性粒(细胞)?(%|(百分(比|数|率|百)?)|(比(例|率)))?)([^计数浸润]{,10}))',
            5: '(C-?反应蛋白[^；、，。,]{3,10})|(CRP[^；、，。,+＋(东)细胞C蛋白酶]{3,10})',
            6: '(血沉[^；、，。,]{3,22})',
            7: '(降钙素原(定量)?([(]细菌感染[)])?([^；、，。,]{3,22}))|(PCT([^；、，。,+＋(东)细胞C蛋白酶]{3,20}))',
            8: '(降钙素原(定量)?([(]细菌感染[)])?([^；、，。,]{3,22}))|(PCT([^；、，。,+＋(东)细胞C蛋白酶]{3,20}))',
            9: '(脂肪酶[^；、，。,]{3,22})|(LIPA[^；、，。,+＋(东)细胞C蛋白酶]{3,20})',
            10: '(淀粉酶[^；、，。,]{3,22})|(AMY[^；、，。,+＋(东)细胞C蛋白酶]{3,20})',
            11: '(天冬氨酸氨基转移酶[^；、，。,]{3,22})|(AST[^；、，。,+＋(东)细胞C蛋白酶电解质]{3,20})',
            12: '(丙氨酸氨基转移酶[^；、，。,]{3,22})|(ALT[^；、，。,+＋(东)细胞C蛋白酶电解质]{3,20})',
            13: '(碱性磷酸酶[^；、，。,]{3,22})|(ALP[^；、，。,+＋(东)细胞C蛋白酶电解质]{3,20})',
            14: '(谷氨酰转移酶[^；、，。,]{3,22})|(GGT[^；、，。,+＋(东)细胞C蛋白酶电解质]{3,20})',
            15: '(绒毛膜促性腺激素[^；、，。,]{3,22})|(HCG[^；、，。,+＋(东)细胞C蛋白酶]{3,20})',
            16: '(总胆红素[^；、，。,]{3,22})|(TB(IL)?[^；、，。,+＋(东)细胞C蛋白酶]{3,20})',
            17: '尿.*红细胞([^酶。]{2,10})',
            18: '尿.*红细胞([^酶。]{2,10})'
            # 16: '(红细胞[^；、，。,单位浓度比积体积含量计数(]{3,22}个/L)'  # 此处是血液的，不是尿的
        }

        self.rule_val_regex = '((增高)|(升高)|(较高)|(降低)|(下降)|(减少)|(偏低)|(异常)|(正常)|(阴性)|(（-）)|(阳性)|(（\+）)|(↑)|(↓))|([0-9]{1,}\.?[0-9]{0,3})'

        self.stack_regex = {
            '血常规': '血常规[^。0-9酶]{,20}((异常)|(正常)|(阴性)|(（-）)|(阳性)|(（\+）))',
            '胰腺炎生化': '胰腺炎?生化[^。0-9酶]{,20}((增高)|(升高)|(较高)|(降低)|(下降)|(减少)|(偏低)|(异常)|(正常)|(阴性)|(（-）)|(阳性)|(（\+）)|(↑)|(↓))',
            '肝功能': '肝肾?功能[^。常规]{3,40}((增高)|(升高)|(较高)|(降低)|(下降)|(减少)|(偏低)|(异常)|(正常)|(阴性)|(（-）)|(损伤)|(不全))',
            '尿常规': '尿常规[^。0-9酶]{,20}((异常)|(正常)|(阴性)|(（-）)|(阳性)|(（\+）))'
        }

        self.stack_rules = {
            '血常规': [4, 1, 2, 3],
            '胰腺炎生化': [9, 10],
            '肝功能': [11, 12],
            '尿常规': [17, 18]
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


    def get_rule_label_results(self, rule_results):
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


    def assess_val(self, text, val, rule=None):
        """
        评定val是正向负向还是数字，如果是数字，则与规则进行比较
        """
        if val in ['增高', '升高', '较高', '异常', '阳性', '损伤', '不全']:
            bNeg, bNeg_match = self.check_neg_word(text, val)
            if bNeg:
                return 0, val, bNeg_match, text
            else:
                return 1, val, None, text
        elif val in ['降低', '下降', '减少', '偏低', '正常', '阴性']:
            bNeg, bNeg_match = self.check_neg_word(text, val)
            if bNeg:
                return 1, val, bNeg_match, text
            else:
                return 0, val, None, text
        elif val in ['（-）', '↓']:
            return 0, val, None, text
        elif val in ['（+）', '↑']:
            return 1, val, None, text
        else:
            val = self.utils.format_num(val)
            if val == '':
                return 0, val, None, text
            else:
                val = float(val)
                assert rule, '规则对象为空'
                if rule[2] == 'gt':
                    if val > rule[3]:
                        return 1, val, None, text
                    else:
                        return 0, val, None, text
                else:
                    if val < rule[3]:
                        return 1, val, None, text
                    else:
                        return 0, val, None, text


    def process_rule_str_regex(self, texts, rule_results):
        """
        从自由文本中提取实验室数据特征。
        输入：
            文本数组
        输出：
            所有特征提取结果
            {rule_id: 0,1,2}
        """
        for text in texts:
            for rule_id in self.rules.keys():
                rule = self.rules[rule_id]
                regex = self.rule_str_regex[rule_id]
                if self.debug:
                    print(rule_id, regex)
                for match in re.finditer(regex, text):
                    groups = match.groups()
                    for i in range(len(groups) - 1, -1, -1):
                        if groups[i]:
                            val = re.search(self.rule_val_regex, groups[i])
                            if self.debug:
                                print('group selected: %s, val: %s' % (groups[i], val))
                            if val:
                                rt, _, _, _ = self.assess_val(groups[i], val.group(0), rule)
                                if self.debug:
                                    print('assesss value: %s' % rt)
                                if rule_results[rule_id] == 2 or rt == 1:
                                    rule_results[rule_id] = rt
                                break
                    break

        return rule_results


    def process_stack_regex(self, texts, rule_results):
        """
        从自由文本中组套特征，并赋值给实验室特征。
        输入：
            文本数组
        输出：
            所有特征提取结果
            {rule_id: 0,1,2}
        """
        stack_results = {stack: 2 for stack in self.stack_regex.keys()}
        for text in texts:
            for stack in self.stack_regex.keys():
                regex = self.stack_regex[stack]
                for match in re.finditer(regex, text):
                    rt, _, _, _ = self.assess_val(match.group(0), match.group(1), None)
                    if stack_results[stack] == 2 or rt == 1:
                        stack_results[stack] = rt
                    break

        # 组套结果赋值给正则结果
        for stack in self.stack_regex.keys():
            rt = stack_results[stack]
            for rule_id in self.stack_rules[stack]:
                if rule_results[rule_id] == 2 or rt == 1:
                    rule_results[rule_id] = rt

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
        rule_results = {k:2 for k in self.rules.keys()}

        # rule_str_regex
        rule_results = self.process_rule_str_regex(texts, rule_results)

        # stack regex
        rule_results = self.process_stack_regex(texts, rule_results)

        return rule_results





#

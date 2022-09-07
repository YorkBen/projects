"""
超声、放射数据特征抽取和疾病预测模块
"""
import json
import re

inner_neg = '[^，；、。无未不]{,4}'
inner_neg_l = '[^，；、。无未不]{,8}'
inner = '[^，；、。]*?'
regex_suspect = '((？)|(?)|(怀疑)|(待排)|(可能))'

class InspExtractor:
    def __init__(self, json_file):
        # 加载json数据
        json_data = ''
        with open(json_file) as f:
            json_data = json.load(f, strict=False)

        # 构建字典
        json_dict = {}
        for item in json_data:
            json_dict[item["医保编号"]] = item

        self.json_dict = json_dict

        # 疾病对应的正则
        self.diease_regex = {
            '急性阑尾炎': '阑尾.*' + inner_neg + '((炎)|(脓肿)|(穿孔))',
            '急性胰腺炎': '胰腺.*' + inner_neg + '((炎)|(脓肿)|(穿孔))',
            '肠梗阻': '肠梗阻',
            '异位妊娠': '宫外孕',
            '急性胆管炎': '胆管炎',
            '急性胆囊炎': '胆囊炎',
            '上尿路结石': '((肾)|(输尿管)|(膀胱))' + inner_neg + '结石',
            '消化性溃疡穿孔': '穿孔'
        }

        # 疾病待排对应的正则
        self.diease_suspect_regex = {
            '急性胆管炎': '胆管炎' + inner + '((？)|(?)|(怀疑)|(待排)|(可能)|(硬化性))'
        }


    def extract_jielun(self, record):
        """
        从放射和超声数据里面，提取所有的结论文本。
        """
        txt_dict = {}
        for key in ['放射', '超声']:
            if key in record:
                for item in record[key]:
                    for item2 in item['数据']
                        arr = item2.split(',')
                        if arr[0] == '' or arr[1] == '' or arr[-1] == '':
                            continue
                        if not arr[0] in txt_dict:
                            txt_dict[arr[0]] = {}
                        if not arr[1] in txt_dict[arr[0]]:
                            txt_dict[arr[0]][arr[1]] = []
                        txt_dict[arr[0]][arr[1]].append(arr[-1])

        return txt_dict


    def process(self, mrno, insp_type, body_part):
        """
        mrno：病历号
        insp_type：检验类型，超声（US），CT,CR,DR,MR
        body_part：检查部位
        """
        record = self.json_dict[mrno]
        txt_dict = extract_jielun(record)
        txt_arr = []
        if insp_type in txt_dict and body_part in txt_dict[insp_type]:
            txt_arr = txt_dict[insp_type][body_part]
        result = {}
        for diease in self.diease_regex.keys():
            result[diease] = 0
            for txt in txt_arr:
                if re.search(self.diease_regex[diease], txt):
                    result[diease] = 1
                if diease in self.diease_suspect_regex and re.search(self.diease_suspect_regex[diease], txt):
                    result[diease] = 3
                elif re.search(self.diease_regex[diease] + inner + regex_suspect, txt):
                    result[diease] = 3

        return result




#

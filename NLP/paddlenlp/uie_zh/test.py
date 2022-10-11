from paddlenlp import Taskflow
from pprint import pprint

# schema = {'恶心呕吐':['有', '无'], '压痛':['有', '无'], '肾脏叩击痛': ['有', '无']}
# schema = ['恶心呕吐', '压痛', '肾脏叩击痛', '症状']
# schema = {'症状': ['部位', '否定词', '性质']}
schema = ['体温', '脉搏', '血压', '呼吸', '发育', '营养']
ie = Taskflow('information_extraction', schema=schema, model="uie-medical-base", device_id=-1)
# ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best', device_id=-1)

text = '患者4年前发现血糖升高并诊断为2型糖尿病，口服格列美脲片，阿卡拜糖片降糖治疗，未规律监测血糖。近3天无明显诱因出现恶心、呕吐、腹泻症状，偶伴头昏、乏力，无发热、手抖等症状。为进一步诊治，遂来我院本院区查2020-7-30电解质：血钾5.36mmol/L，Na131mmol/l，Cl95mmol/l。心梗三项正常。淀粉酶32U/l，脂肪酶23U/L。尿酮体3+，尿糖3+。胸部CT提示炎性结节。轻度脂肪肝。门诊以“糖尿病酮症酸中毒”收入我科。'
text = '呕吐近1月，加重1周'
text = '患者自入院以来，无恶心、呕吐，无头痛、头晕，有发热、寒战，神志清'
text = '体温: 36.8℃, 脉搏: 82次/分（ 规则）, 呼吸: 17次/分（ 规则）, 血压: 114/72mmhg, 一般情况: 发育: 正常, 营养: 良好, 表情: 自如'
pprint(ie(text))

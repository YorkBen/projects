# pip install openpyxl
import re
import os
import argparse

def load_regex():
    lines = []
    for filename in ['放射', '超声', '检验_腹痛']:
        with open(r'%s.txt' % filename) as f:
            for l in f.readlines():
                lines.append(l.strip())

    return '((' + ')|('.join(lines) + '))'

if __name__ == '__main__':
    # 参数
    regex = load_regex()
    s = r'患者半月前无明显诱因出现腹痛，主要见于剑突下及右上腹疼痛，偶有肩背部放射痛，与饮食无明显相关，每次发作持续数分钟不等，可自行缓解，不伴发热、咳嗽，不伴反酸、烧心、嗳气，不伴恶心、呕吐，无头晕、心慌、气促等不适。5天前，皮肤逐渐黄染，伴尿黄，进行性加重，不伴皮肤瘙痒。为求诊治，来我院求诊，急查血常规提示WBC11.59X109个/L，肝功能提示ALT314U/L、AST339U/L，TBIL81.2umol/L、DBIL59.7umol/L，予以抗炎、护肝、补液等治疗，请消化内科会诊后收入我科。起病以来，精神、睡眠尚可，食欲减退贵，大小便如常，体力体重无明显改变。'
    print(re.search(regex, s))

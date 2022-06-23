import sys

sys.path.append('..')

from FeatureExtraction.utils import *

"""
计算疾病典型特征匹配到病症的数量
"""

diease_features = [
    {'name': '急性阑尾炎DB10', 'features': ['厌食', '恶心呕吐', '右下腹疼痛', '转移性右下腹疼痛']},
    {'name': '急性胰腺炎DC31', 'features': ['上腹痛', '恶心呕吐', '放射痛']},
    {'name': '肠梗阻DA91', 'features': ['绞痛', '腹胀', '停止排便排气', '恶心呕吐']},
    {'name': '宫外孕JA01', 'features': ['停经', '阴道流血']},
    {'name': '胆石症/胆道炎症DC11', 'features': ['镇痛药可缓解', '恶心呕吐', '上腹痛', '放射痛']},
    {'name': '泌尿系统结石/泌尿系统炎症GB70', 'features': ['腰痛', '排尿改变', '坐立不安', '血尿', '疼痛放射']}
]

#########复合特征########
# 放射痛
# 腰背放射, 右下腹, 右上腹, 左下腹, 下腹
def check_fst(df, rn):
    vc = df[['腰背放射', '右下腹', '右上腹', '左下腹', '下腹']].iloc[rn].value_counts()
    if '1' in vc.index:
        return True
    else:
        return False

# 疼痛放射
# 右下腹，左下腹，右上腹，大腿
def check_ttfs(df, rn):
    vc = df[['右下腹', '右上腹', '左下腹', '大腿']].iloc[rn].value_counts()
    if '1' in vc.index:
        return True
    else:
        return False

# 检查单个特征
def check_single(df, rn, fn):
    if df[fn].iloc[rn] == '1':
        return True
    else:
        return False

df = load_grid('data.txt', separator='	')
results = []
for rn in range(0, len(df.index)):
    rrow = []
    for diease in diease_features:
        ct = 0
        for feature in diease['features']:
            if feature == '放射痛':
                if check_fst(df, rn):
                    ct = ct + 1
            elif feature == '疼痛放射':
                if check_ttfs(df, rn):
                    ct = ct + 1
            else:
                if check_single(df, rn, feature):
                    ct = ct + 1
        rrow.append(ct)
    results.append(rrow)

# 按固定列名来存储
# write_columns(results, [diease['name'] for diease in diease_features], 'result_1.txt')

# 每行按照数量排序后输出
results2 = []
cols = [diease['name'] for diease in diease_features]
for r in results:
    rrnew = []
    for name, ct in zip(cols, r):
        rrnew.append((name, ct))
    results2.append(sorted(rrnew, key=lambda x: x[1], reverse=True))

results = [[r2[0] + ':' + str(r2[1]) for r2 in r1] for r1 in results2]
write_columns(results, [''], 'result_2.txt')

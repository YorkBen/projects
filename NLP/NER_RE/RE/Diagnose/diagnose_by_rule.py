import argparse
import sys

sys.path.append('../Lib')

from MRRecordUtil import *

feature_type_names = {
    '临床表现':['放射痛（右肩、肩胛和背部）', '上腹痛', '厌食', '腹泻', '头晕', '黄疸', '盆腔疼痛', '迅速波及全腹', '坐立不安', '黑便', '右下腹疼痛', '转移性右下腹疼痛', '中腹痛',
            '右上腹痛', '放射痛（放射到同侧半腹或背部）', '持续性腹痛', '仰卧加重，坐位缓解', '阵发性疼痛', '突然发作', '绞痛', '剧烈腹痛', '恶心呕吐', '腹胀', '排便排气减少',
            '停经', '阴道流血', '排尿改变', '放射痛（侧腹、腹股沟、睾丸或大阴唇）', '腰/侧腹痛', '性别', '年龄'],

    '病史特征':['阑尾炎家族史阳性', '腹部手术史', '胆结石病史', '酗酒', '美沙拉明、速尿、氯沙坦、6-巯基嘌呤或硫唑嘌呤、异烟肼、袢利尿剂和去羟肌苷使用史', '高钙血症病史', '高甘油三酯血症病史',
            '病毒(腮腺炎、柯萨奇病毒、巨细胞病毒、水痘、单纯疱疹病毒、HIV)、细菌(支原体、军团菌、钩端螺旋体、沙门氏菌)、寄生虫(弓形虫、隐孢子虫、蛔虫)和真菌(曲霉)',
            'PRSS1、SPINK1、CFTR、CASR、CTRC基因突变', '自身免疫性疾病', '内镜逆行胰胆管造影、EUS与FNA、主动脉手术胰腺切除术史', '胰腺肿瘤、囊肿病史', '腹部肿瘤史',
            '疝气或疝气修复史', '炎症性肠病病史', '子宫内膜异位症病史', '憩室炎病史', '阑尾炎病史', '肠系膜缺血病史', '异物摄入史', '钩虫感染史', '非甾体抗炎药，氯化钾肠溶片使用史', '输卵管手术史',
            '不孕史', '既往生殖器感染史', '既往流产史（包括人工流产）', '吸烟史', '年龄35岁及以上', '宫内节育器使用史(超过2年)', '既往异位妊娠史', '肥胖',
            '输卵管积水病史', '杀精剂接触史', '饱餐、进食油腻食物史', '结节病', '尿路感染史', '卵巢旁囊肿病史', '禁食史', '胃泌素瘤', 'CXCR1低表达', '畸胎瘤病史', '类固醇使用史',
            '环丙沙星、三硅酸镁、磺胺药物、氨苯蝶啶、茚地那韦、愈创甘油醚、麻黄碱、袢利尿剂（呋塞米）、碳酸酐酶抑制剂、泻药（开塞露）、阿昔洛韦、环利尿剂、乙酰唑胺、茶碱、糖皮质激素（泼尼松）、噻嗪、水杨酸、丙磺舒、别嘌呤醇服用史',
            '重症监护室（应激性溃疡）', '消化道溃疡病史', '尿石症家族史', '尿石症病史', '痛风病史', '高血压病史', '高盐摄入史', '高草酸饮食', '糖尿病病史', '饮水较少', '超过一个性伴侣',
            '新的性伴侣', '低钙饮食', '肿瘤化疗史', 'HP感染史', '盆腔炎病史', '克罗恩病病史', '原发性甲状旁腺功能亢进', '甲亢病史', '二乙基己烯雌酚', '多发性骨髓瘤病史', '肾小管酸中毒病史',
            '短肠综合征病史', '减肥手术史', '慢性肾病病史'],

    '体格检查':['下腹压痛', '子宫压痛', '附件区压痛', '肾脏叩击痛', '触及腹部囊肿', '肠鸣音消失', '宫颈刺激', '板状腹', '可触及腹部肿块（肿大胆囊）', '腹部叩诊鼓音', '肠鸣音亢进（病程早期）',
            '肠鸣音减弱（病程晚期）', '昏迷', '体温升高', '脉搏显著加快', '血压下降', '反跳痛或肌紧张', '墨菲征Murphy征', '附件肿块'],

    '实验室':['全血_白细胞_大于_9.5', '全血_白细胞_小于_4', '全血_白细胞_大于_10', '全血_中性粒细胞%_大于_75', '全血_C反应蛋白_大于_10', '全血_血沉_大于_26',
           '全血_降钙素原_大于_0.1', '血清_降钙素原_大于_0.1', '血清_脂肪酶_大于_900', '血清_淀粉酶_大于_405', '血清_天冬氨酸氨基转移酶_大于_75',
           '血清_丙氨酸氨基转移酶_大于_60', '血清_碱性磷酸酶_大于_187.5', '血清_γ-谷氨酰转移酶_大于_90', '血清_β-绒毛膜促性腺激素_大于_10', '血清_总胆红素_大于_22',
           '尿液_红细胞_大于_28', '尿液_白细胞_大于_17', '血清_C反应蛋白_大于_5'],

    '影像学':['超声_急性阑尾炎', '超声_急性胰腺炎', '超声_肠梗阻', '超声_异位妊娠', '超声_急性胆管炎', '超声_急性胆囊炎', '超声_上尿路结石', '超声_消化道穿孔', '超声_卵巢囊肿',
           'CT_急性阑尾炎', 'CT_急性胰腺炎', 'CT_肠梗阻', 'CT_异位妊娠', 'CT_急性胆管炎', 'CT_急性胆囊炎', 'CT_上尿路结石', 'CT_消化道穿孔', 'CT_卵巢囊肿',
           'MR_急性阑尾炎', 'MR_急性胰腺炎', 'MR_肠梗阻', 'MR_异位妊娠', 'MR_急性胆管炎', 'MR_急性胆囊炎', 'MR_上尿路结石', 'MR_消化道穿孔', 'MR_卵巢囊肿',
           'DR_急性阑尾炎', 'DR_急性胰腺炎', 'DR_肠梗阻', 'DR_异位妊娠', 'DR_急性胆管炎', 'DR_急性胆囊炎', 'DR_上尿路结石', 'DR_消化道穿孔', 'DR_卵巢囊肿']

}


def check_diease_diagnosis(feature_clinic, feature_history, feature_health, feature_lab, feature_img):
    """
    检查疾病确诊
    """
    result = [0 for i in range(10)]
    # 急性阑尾炎
    if feature_img[feature_type_names['影像学'].index('CT_急性阑尾炎')] == 1:
        result[0] = 1

    # 急性胰腺炎
    c1, c2, c3 = 0, 0, 0
    if feature_clinic[feature_type_names['临床表现'].index('上腹痛')] == 1:
        c1 = 1
    if feature_lab[feature_type_names['实验室'].index('血清_淀粉酶_大于_405')] == 1:
        c2 = 1
    if feature_img[feature_type_names['影像学'].index('超声_急性胰腺炎')] == 1 or \
        feature_img[feature_type_names['影像学'].index('CT_急性胰腺炎')] == 1 or \
        feature_img[feature_type_names['影像学'].index('MR_急性胰腺炎')] == 1:
        c3 = 1
    if c1 + c2 + c3 >= 2:
        result[1] = 1

    # 肠梗阻
    if feature_img[feature_type_names['影像学'].index('DR_肠梗阻')] == 1 or \
        feature_img[feature_type_names['影像学'].index('CT_肠梗阻')] == 1:
        result[2] = 1

    # 异位妊娠
    if feature_img[feature_type_names['影像学'].index('超声_异位妊娠')] == 1:
        result[3] = 1

    # 急性胆管炎
    c1, c2, c3 = 0, 0, 0
    if feature_health[feature_type_names['体格检查'].index('体温升高')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('全血_白细胞_大于_10')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('全血_中性粒细胞%_大于_75')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('全血_C反应蛋白_大于_10')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('全血_血沉_大于_26')] == 1:
        c1 = 1
    if feature_clinic[feature_type_names['临床表现'].index('黄疸')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('血清_天冬氨酸氨基转移酶_大于_75')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('血清_丙氨酸氨基转移酶_大于_60')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('血清_碱性磷酸酶_大于_187.5')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('血清_γ-谷氨酰转移酶_大于_90')] == 1:
        c2 = 1
    if feature_img[feature_type_names['影像学'].index('超声_急性胆管炎')] == 1 or \
        feature_img[feature_type_names['影像学'].index('CT_急性胆管炎')] == 1 or \
        feature_img[feature_type_names['影像学'].index('MR_急性胆管炎')] == 1:
        c3 = 1
    # print('急性胆管炎：c1: %s, c2: %s, c3:%s' % (c1, c2, c3))
    if c1 * c2 * c3 >= 1:
        result[4] = 1

    # 急性胆囊炎
    c1, c2, c3 = 0, 0, 0
    if feature_health[feature_type_names['体格检查'].index('墨菲征Murphy征')] == 1:
        c1 = 1
    if feature_health[feature_type_names['体格检查'].index('体温升高')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('全血_C反应蛋白_大于_10')] == 1 \
        or feature_lab[feature_type_names['实验室'].index('全血_白细胞_大于_10')] == 1:
        c2 = 1
    if feature_img[feature_type_names['影像学'].index('超声_急性胆囊炎')] == 1 or \
        feature_img[feature_type_names['影像学'].index('CT_急性胆囊炎')] == 1 or \
        feature_img[feature_type_names['影像学'].index('MR_急性胆囊炎')] == 1:
        c3 = 1

    if c1 * c2 * c3 >= 1:
        result[5] = 1

    # 上尿路结石
    if feature_img[feature_type_names['影像学'].index('超声_上尿路结石')] == 1 or \
        feature_img[feature_type_names['影像学'].index('CT_上尿路结石')] == 1 or \
        feature_img[feature_type_names['影像学'].index('DR_上尿路结石')] == 1:
        result[6] = 1

    # 卵巢囊肿破裂, 卵巢囊肿扭转
    if feature_img[feature_type_names['影像学'].index('超声_卵巢囊肿')] == 1 or \
        feature_img[feature_type_names['影像学'].index('CT_卵巢囊肿')] == 1 or \
        feature_img[feature_type_names['影像学'].index('MR_卵巢囊肿')] == 1 or \
        feature_img[feature_type_names['影像学'].index('DR_卵巢囊肿')] == 1:
        result[7] = 1
        result[8] = 1

    # 消化道穿孔
    if feature_img[feature_type_names['影像学'].index('CT_消化道穿孔')] == 1 or \
        feature_img[feature_type_names['影像学'].index('DR_消化道穿孔')] == 1:
        result[9] = 1


    return result


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-i', type=str, default='data/', help='特征文件')
    parser.add_argument('-o', type=str, default='data/diagnose_by_rule_output.txt', help='输出结果')
    args = parser.parse_args()

    input = args.i
    output = args.o

    results = []
    with open(input) as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                continue

            arr = line.strip().split('	')
            r = arr[:3]

            start_col, end_col = 3, 3 + len(feature_type_names['临床表现'])
            feature_clinic = [float(e) for e in arr[start_col:end_col]]

            start_col = end_col
            end_col = start_col + len(feature_type_names['病史特征'])
            feature_history = [float(e) for e in arr[start_col:end_col]]

            start_col = end_col
            end_col = start_col + len(feature_type_names['体格检查'])
            feature_health = [float(e) for e in arr[start_col:end_col]]

            start_col = end_col
            end_col = start_col + len(feature_type_names['实验室'])
            feature_lab = [float(e) for e in arr[start_col:end_col]]

            start_col = end_col
            end_col = start_col + len(feature_type_names['影像学'])
            feature_img = [float(e) for e in arr[start_col:end_col]]

            r.extend(check_diease_diagnosis(feature_clinic, feature_history, feature_health, feature_lab, feature_img))
            results.append(r)


    with open(output, 'w') as f:
        f.write('医保编号	入院时间	诊断编号	急性阑尾炎	急性胰腺炎	肠梗阻	异位妊娠	急性胆管炎	急性胆囊炎	上尿路结石	卵巢囊肿破裂	卵巢囊肿扭转	消化道穿孔\n')
        for r in results:
            arr = [str(e) for e in r]
            f.write('%s\n' % '	'.join(arr))

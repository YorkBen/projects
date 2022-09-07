"""
临床正则提取类
"""
import sys
import json
import re
import logging
import os
import argparse
import copy
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, Border, Side, PatternFill, colors, Alignment

sys.path.append('../FeatureExtracModel')
sys.path.append('../Lib')
from RegexBase import RegexBase
from MRRecordUtil import *
from RegexUtil import RegexUtil

logging.basicConfig(level=logging.DEBUG)

class ClinicRule(RegexBase):
    def __init__(self, type='腹痛', postfix='4335'):
        super(ClinicRule, self).__init__()

        self.type = type
        self.postfix = postfix

        self.utils = RegexUtil()
        self.hl_font1 = Font(name="Arial", size=14, color="00FF0000")
        self.hl_font2 = Font(name="Arial", size=14, color="0000FF00")

        # 疾病对应的正则
        self.features = [
            ### 临床表现
            {'id':'FSTX1', 'name': '放射痛（右肩、肩胛和背部）', 'src': ['现病史'], 'type': 'model'},   # model
            {'id':'SFTX1', 'name': '上腹痛', 'src': ['现病史'], 'type': 'model'},   # model
            {'id':'YSX1', 'name': '厌食', 'src': ['现病史'], 'type': 'model'},   # model
            {'id':'FXX1', 'name': '腹泻', 'src': ['现病史'], 'type': 'model'},   # model
            {'id':'TYX1', 'name': '头晕', 'src': ['现病史'], 'regex':'(头晕)|(晕倒)',  'default': 2},
            {'id':'HDX1', 'name': '黄疸', 'src': ['现病史'], 'regex':'(黄疸)|(黄染)|(发黄)', 'default': 2},
            {'id':'PQTTX1', 'name': '盆腔疼痛', 'src': ['现病史'], 'type': 'model'},   # model
            {'id':'BJQX1', 'name': '迅速波及全腹', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'ZLBX1', 'name': '坐立不安', 'src': ['现病史'], 'regex':'坐立不安', 'default': 2},
            {'id':'HBX1', 'name': '黑便', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'YXFTX1', 'name': '右下腹疼痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'ZYXTX1', 'name': '转移性右下腹疼痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'ZFTX1', 'name': '中腹痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'YSFTX1', 'name': '右上腹痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'FSDFX1', 'name': '放射痛（放射到同侧半腹或背部）', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'CXFTX1', 'name': '持续性腹痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'YHJZX1', 'name': '仰卧加重，坐位缓解', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'ZFXTX1', 'name': '阵发性疼痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'TRFZX1', 'name': '突然发作', 'src': ['现病史'], 'regex':'(突发)|(突然)|(突感)|(突起)|(突生)', 'default': 2},
            {'id':'JTX1', 'name': '绞痛', 'src': ['现病史'], 'regex':'绞痛', 'default': 2},
            {'id':'JLFTX1', 'name': '剧烈腹痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'EXOTX1', 'name': '恶心呕吐', 'src': ['现病史'], 'regex':'(恶心)|(呕吐)|呕|吐',  'default': 2},
            {'id':'FZX1', 'name': '腹胀', 'src': ['现病史', '查体'], 'regex':'([腹饱]胀)|(胀[痛疼])|(腹' + self.inner_neg + '胀)',  'default': 2},
            {'id':'TZPBX1', 'name': '停止排便排气', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'TJX1', 'name': '停经', 'src': ['现病史', '月经史'], 'regex': '停经', 'regex_yjzq':'[^末次]{2}月经.*?(周期)?(规律|则)?.*?((([0-9]-)?[0-9]/)?([1-9][0-9]-)?([1-9][0-9]))',
                    'regex_lmp': '((末次月经)|(lmp))[^，。；,]*', 'regex_pmp': 'pmp', 'default': 2},
            {'id':'YDLXX1', 'name': '阴道流血', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'PNGBX1', 'name': '排尿改变', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'FTSCFX1', 'name': '放射痛（侧腹、腹股沟、睾丸或大阴唇）', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'YCFTX1', 'name': '腰、侧腹痛', 'src': ['现病史'], 'type': 'model'},  # model
            {'id':'SEX', 'name': '性别', 'src': ['性别'], 'default': 1}, # 0 男，1 女
            {'id':'AGE', 'name': '年龄', 'src': ['年龄'], 'default': 38.8}, # 38.8 中国人平均年龄


            ### 病史特征
            {'id':'LWYJJ1', 'name': '阑尾炎家族史阳性', 'src': ['家族史'], 'regex':'阑尾((炎)|(脓肿)|(切除))', 'default': 0},
            {'id':'FBSS1', 'name': '腹部手术史', 'src': ['既往史', '手术外伤史'], 'regex':'腹' + self.inner_neg + '手?术', 'default': 0, 'type': 'model'},  # model
            {'id':'DJSJ1', 'name': '胆结石病史', 'src': ['既往史', '手术外伤史'], 'regex':'(胆' + self.inner_neg_x + '石)|(胆囊炎)', 'default': 0},
            {'id':'XJG1', 'name': '酗酒', 'src': ['个人史', '既往史'], 'regex':'酗酒', 'default': 0},
            {'id':'MSLY1', 'name': '美沙拉明、速尿、氯沙坦、6-巯基嘌呤或硫唑嘌呤、异烟肼、袢利尿剂和去羟肌苷使用史', 'src': ['既往史'],  'regex':'(美沙拉明)|(速尿)|(呋塞米)|(氯沙坦)|(科素亚)|(洛沙坦)|(罗沙藤)|(6-巯基嘌呤)|(乐疾宁)|(巯嘌呤)|(巯基嘌呤)|(硫唑嘌呤)|(依木兰)|(异烟肼)|(雷米封)|(袢利尿剂)|(去羟肌苷)', 'default': 0},
            {'id':'GGXZJ1', 'name': '高钙血症病史', 'src': ['既往史'], 'regex':'高钙血症', 'default': 0},
            {'id':'GYSZJ1', 'name': '高甘油三酯血症病史', 'src': ['既往史'], 'regex':'(高脂血症?)|(高血脂症?)|(((甘油三酯)|(血脂)|(胆固醇))' + self.inner_neg + '高)', 'default': 0},
            {'id':'BDJ1', 'name': '病毒(腮腺炎、柯萨奇病毒、巨细胞病毒、水痘、单纯疱疹病毒、HIV)、细菌(支原体、军团菌、钩端螺旋体、沙门氏菌)、寄生虫(弓形虫、隐孢子虫、蛔虫)和真菌(曲霉)', 'src': ['既往史'], 'regex':'(腮腺炎)|(柯萨奇病毒)|(巨细胞病毒)|(水痘)|(疱疹)|(HIV)|(人[类体]?免疫缺陷病毒)|(艾滋病)|(支原体)|(军团菌)|(钩端螺旋体)|(沙门氏菌)|(弓形虫)|(隐孢子虫)|(蛔虫)|(真菌)|(曲霉)', 'default': 0},
            {'id':'JYTBJ1', 'name': 'PRSS1、SPINK1、CFTR、CASR、CTRC基因突变', 'src': ['既往史'], 'regex':'(PRSS1)|(SPINK1)|(CFTR)|(CASR)|(CTRC)', 'default': 0}, # 0
            {'id':'MYXJ1', 'name': '自身免疫性疾病', 'src': ['既往史'], 'regex':'', 'default': 0, 'type': 'model'},   # model
            {'id':'DGZYJ1', 'name': '内镜逆行胰胆管造影、EUS与FNA、主动脉手术胰腺切除术史', 'src': ['手术外伤史'],  'regex':'(胰腺)|(ERCP)', 'default': 0},
            {'id':'YXZLJ1', 'name': '胰腺肿瘤、囊肿病史', 'src': ['既往史'], 'regex':'胰腺.*((肿瘤)|(囊肿))', 'default': 0, 'type': 'model'},   # model
            {'id':'FBZJ1', 'name': '腹部肿瘤史', 'src': ['既往史', '手术外伤史'], 'regex':'腹部' + self.inner_neg + '肿?瘤', 'default': 0, 'type': 'model'},    # model
            {'id':'SQJ1', 'name': '疝气或疝气修复史', 'src': ['既往史', '手术外伤史'], 'regex':'[^食管裂孔外膈]{,4}疝', 'default': 0},
            {'id':'YCBJ1', 'name': '炎症性肠病病史', 'src': ['既往史'], 'regex':'(炎症性肠病)|(溃疡性结肠炎)|(溃结)|(克罗恩)', 'default': 0},
            {'id':'NMYWJ1', 'name': '子宫内膜异位症病史', 'src': ['既往史', '手术外伤史'], 'regex':'((子宫内膜异位)|(内异症))', 'default': 0},
            {'id':'QSYJ1', 'name': '憩室炎病史', 'src': ['既往史'], 'regex':'憩室炎', 'default': 0},
            {'id':'LWYJ1', 'name': '阑尾炎手术史', 'src': ['既往史', '手术外伤史'], 'regex':'(阑尾(炎|(阑尾脓肿)|(切除)))', 'default': 0},
            {'id':'CXMJ1', 'name': '肠系膜缺血病史', 'src': ['既往史'], 'regex':'肠系膜' + self.inner_neg + '缺血', 'default': 0},
            {'id':'YWJ1', 'name': '异物摄入史', 'src': ['既往史', '现病史'], 'regex':'异物', 'default': 0},
            {'id':'GCGJ1', 'name': '钩虫感染病史', 'src': ['既往史'], 'regex':'钩虫', 'default': 0},
            {'id':'FZTJ1', 'name': '非甾体抗炎药，氯化钾肠溶片使用史', 'src': ['既往史'], 'regex':'(非甾体抗炎药)|(阿司匹林)|(布洛芬)|(对乙酰氨基酚)|(吲哚美辛)|(萘普生)|(萘普酮)|(氯芬酸)|(尼美舒利)|(罗非昔布)|(塞来昔布)|(氯化钾)|(补达秀)', 'default': 0},
            {'id':'SLRJ1', 'name': '输卵管手术史', 'src': ['既往史', '手术外伤史'], 'regex':'((输卵管)|(附件))' + self.inner_neg + '((手?术)|(结扎)|(切除))', 'default': 0},
            {'id':'BYHJ1', 'name': '不孕(风险随不孕时间的延长而增加)', 'src': ['婚育史', '既往史', '现病史'], 'regex':'不孕'},
            {'id':'JWSJ1', 'name': '既往生殖器感染', 'src': ['既往史'], 'regex':'', 'default': 0, 'type': 'model'},   # model
            {'id':'LCJ1', 'name': '既往流产（包括人工流产）', 'src': ['婚育史'], 'regex':'流产', 'default': 0},
            {'id':'XYGS1', 'name': '吸烟', 'src': ['个人史', '既往史'], 'regex':'吸烟', 'default': 0},
            {'id':'NL35', 'name': '年龄35岁及以上', 'src': ['年龄'], 'regex':''},
            {'id':'GNJJ1', 'name': '宫内节育器使用(超过2年)', 'src': ['现病史', '既往史', '手术外伤史'], 'regex':'(节育器)|(节育环)', 'default': 0},
            {'id':'RSSJ1', 'name': '既往异位妊娠史', 'src': ['既往史', '手术外伤史'], 'regex':'(((间质部)|(疤痕)|(切口)|(宫角)|(异位)|(输卵管)|(卵巢)|(瘢痕))处?妊娠)|(宫外孕)', 'default': 0},
            {'id':'FPJ1', 'name': '肥胖', 'src': ['既往史'], 'regex':'肥胖', 'default': 0},
            {'id':'SLGJ1', 'name': '输卵管积水', 'src': ['现病史', '既往史'], 'regex':''},
            {'id':'SJJJ1', 'name': '杀精剂接触史', 'src': ['既往史'], 'regex':'壬苯醇醚', 'default': 0},
            {'id':'YNJ1', 'name': '饱餐、进食油腻食物史', 'src': ['现病史', '既往史'], 'regex':'(饱餐)|(油腻)|(油脂)', 'default': 0},
            {'id':'JJJ1', 'name': '结节病病史', 'src': ['既往史'], 'regex':'结节病?', 'default': 0},
            {'id':'NLJ1', 'name': '尿路感染史', 'src': ['既往史'], 'regex':'(肾盂肾炎)|(肾炎)|(输尿管炎)|(膀胱炎)|(尿道炎)', 'default': 0},
            {'id':'LCPJ1', 'name': '卵巢旁囊肿', 'src': ['既往史'], 'regex':'', 'default': 0},
            {'id':'JSJ1', 'name': '禁食', 'src': ['现病史', '既往史'], 'regex':'', 'default': 0},
            {'id':'WBSLJ1', 'name': '胃泌素瘤', 'src': ['既往史'], 'regex':'(胃泌素瘤)|(卓-艾综合征)|(Z-E综合征)', 'default': 0},
            {'id':'CXCRJ1', 'name': 'CXCR1低表达', 'src': ['现病史', '既往史'], 'regex':'CXCR1' + self.inner_neg + '低', 'default': 0},
            {'id':'JTLJ1', 'name': '畸胎瘤病史', 'src': ['现病史', '既往史'], 'regex': '畸胎瘤', 'default': 0},
            {'id':'LGCJ1', 'name': '类固醇', 'src': ['既往史'], 'regex':'(大力补)|(康力龙)|(康复龙)|(睾酮)', 'default': 0},
            {'id':'HBSJ1', 'name': '环丙沙星、三硅酸镁、磺胺药物、氨苯蝶啶、茚地那韦、愈创甘油醚、麻黄碱、袢利尿剂（呋塞米）、碳酸酐酶抑制剂、泻药（开塞露）、阿昔洛韦、环利尿剂、乙酰唑胺、茶碱、糖皮质激素（泼尼松）、噻嗪、水杨酸、丙磺舒、别嘌呤醇服用史', 'src': ['既往史'], 'regex':'(环丙沙星)|(三硅酸镁)|(磺胺)|(苯磺胺)|(氨苯蝶啶)|(茚地那韦)|(愈创甘油醚)|(麻黄碱)|(托拉塞米)|(布美他尼)|(呋塞米)|(乙酰唑胺)|(碳酸酐酶抑制剂)|(泻药)|(开塞露)|(阿昔洛韦)|(环利尿剂)|(乙酰唑胺)|(茶碱)|(泼尼松)|(甲泼尼松龙)|(倍他米松)|(氢化可的松)|(可的松)|(地塞米松)|(噻嗪)|(氢氯噻嗪)|(阿司匹林)|(水杨酸)|(丙磺舒)|(别嘌呤醇)', 'default': 0},
            {'id':'ZZJJ1', 'name': '重症监护室（应激性溃疡）', 'src': ['现病史', '既往史'], 'regex':'应激性溃疡', 'default': 0},
            {'id':'XHDJ1', 'name': '消化道溃疡病史', 'src': ['现病史', '既往史'], 'regex':'(肠溃疡)|(十二指肠球部溃疡)|(胃溃疡)|(消化道溃疡)|(胃窦巨大溃疡)|(胃角巨大溃疡)', 'default': 0},
            {'id':'NJJ1', 'name': '尿石症家族史', 'src': ['家族史'], 'regex':'(肾结石)|(肾小结石)|(肾多发小结石)|(肾点状结石)|(输尿管结石)|(输尿管上段结石)', 'default': 0},
            {'id':'NSZJ1', 'name': '尿石症病史', 'src': ['既往史'], 'regex':'((肾|(输尿管))镜?' + self.inner_neg + '(结|碎|取)石术?)', 'default': 0},
            {'id':'TFJ1', 'name': '痛风病史', 'src': ['既往史'], 'regex':'痛风', 'default': 0},
            {'id':'GXYJ1', 'name': '高血压病史', 'src': ['既往史'], 'regex':'高血压', 'default': 0},
            {'id':'GYJ1', 'name': '高盐饮食', 'src': ['现病史', '既往史'], 'regex':'高盐', 'default': 0},
            {'id':'GCJ1', 'name': '高草酸饮食', 'src': ['现病史', '既往史'], 'regex':'高草酸', 'default': 0},
            # # {'id':'GDJ1', 'name': '高动物蛋白饮食', 'type': None} # None
            {'id':'TNBJ1', 'name': '糖尿病病史', 'src': ['既往史'], 'regex':'糖尿病', 'default': 0},
            {'id':'YSSJ1', 'name': '饮水较少', 'src': ['现病史', '既往史'], 'regex':'(饮水少)|(饮水较少)|(少饮水)', 'default': 0},
            {'id':'YGXJ1', 'name': '超过一个性伴侣', 'src': ['现病史', '既往史'], 'regex':'(冶游)|(不洁性交)', 'default': 0},
            {'id':'XBLJ1', 'name': '新的性伴侣', 'src': ['现病史', '既往史'], 'regex':'(冶游)|(不洁性交)', 'default': 0},
            {'id':'DGJ1', 'name': '低钙饮食', 'src': ['现病史', '既往史'], 'regex':'低钙饮食', 'default': 0},
            {'id':'HLSJ1', 'name': '肿瘤化疗史', 'src': ['既往史', '手术外伤史'], 'regex':'化疗', 'default': 0, 'type': 'model'},   # model
            {'id':'HPJ1', 'name': 'Hp感染史', 'src': ['既往史'], 'regex':'(幽门螺旋杆菌感染)|(HP感染)|(HP（+）)|(HP阳性)', 'default': 0},
            {'id':'PQYJ1', 'name': '盆腔炎病史', 'src': ['既往史'], 'regex':'盆腔炎', 'default': 0},
            {'id':'KLEJ1', 'name': '克罗恩病病史', 'src': ['既往史'], 'regex':'克罗恩病?', 'default': 0},
            {'id':'JZPJ1', 'name': '原发性甲状旁腺功能亢进', 'src': ['既往史'], 'regex':'甲状旁腺功能亢进', 'default': 0},
            {'id':'JKJ1', 'name': '甲亢病史', 'src': ['既往史'], 'regex':'(甲状腺功能亢进)|(甲亢)', 'default': 0},
            {'id':'WYJJ1', 'name': '二乙基己烯雌酚', 'src': ['既往史'], 'regex':'二乙基己烯雌酚', 'default': 0},
            {'id':'DFXJ1', 'name': '多发性骨髓瘤病史', 'src': ['既往史'], 'regex':'多发性骨髓瘤', 'default': 0},
            {'id':'SXGJ1', 'name': '肾小管酸中毒病史', 'src': ['既往史'], 'regex':'肾小管酸中毒', 'default': 0},
            {'id':'DCZJ1', 'name': '短肠综合征', 'src': ['既往史', '手术外伤史'], 'regex':'短肠综合征', 'default': 0},
            {'id':'WRDJ1', 'name': '胃绕道手术史、减肥手术史', 'src': ['既往史', '手术外伤史'], 'regex':'(胃绕道)|(减肥手术)|(缩胃)|(胃缩)', 'default': 0},
            {'id':'MXSJ1', 'name': '慢性肾病病史', 'src': ['既往史'], 'regex':'(尿毒症)|(肾衰)', 'default': 0},

            # # {'id':'XKT1', 'name': '失血性休克', 'src': ['体格检查'], 'regex':'', 'default': 0, 'type': 'model'},   # model

            ### 体格检查
            {'id':'FYTS1', 'name': '下腹压痛', 'src': ['查体'], 'regex':'(([下全]腹)|(脐下))' + self.inner_neg + '压痛', 'negregex': '([下全]腹' + self.inner + '(无|(未见)|(未及))' + self.inner + '压痛)|(上腹' + self.inner + '压痛)', 'default': 0},
            {'id':'ZGYTS1', 'name': '子宫压痛', 'src': ['查体'], 'regex':'子宫' + self.inner_neg_xx3 + '压痛', 'default': 0}, #全是2
            {'id':'FJTS1', 'name': '附件区压痛', 'src': ['查体'], 'regex':'(附件区?' + self.inner_neg_xx3 + '压痛)|(输卵管压痛)|(卵巢压痛)', 'default': 0}, #全是2
            # {'id':'FJYTS1', 'name': '附件压痛', 'src': ['查体'], 'regex':'(附件区' + self.inner_neg + '压痛)|(输卵管压痛)|(卵巢压痛)'}, # 没用了？
            {'id':'SZKTS1', 'name': '肾脏叩击痛', 'src': ['查体'], 'regex':'肾' + self.inner_neg + '叩击?痛', 'negregex': '肾' + self.inner + '(无|(未见)|(未及))' + self.inner + '叩击痛', 'default': 0},
            {'id':'CDNZS1', 'name': '可能触到囊肿', 'src': ['查体'], 'regex':'触?[及到]' + self.inner_neg_x + '囊肿', 'default': 0}, # 全是2
            {'id':'CYXSS1', 'name': '肠鸣音消失', 'src': ['体格检查', '查体'], 'regex':'肠鸣?音((消失)|(未及))', 'default': 0},
            {'id':'GJCJS1', 'name': '宫颈刺激', 'src': ['查体'], 'regex':'宫颈' + self.inner_neg3 + '((刺激)|(举痛)|(摇举痛))', 'default': 0},  # 全是2
            {'id':'BZFS1', 'name': '板状腹', 'src': ['查体'], 'regex':'(板状腹)|(腹肌紧张)|(腹肌强直)|(腹壁紧张)|(腹壁强直)', 'default': 0},
            {'id':'ZDDNT1', 'name': '肿大胆囊', 'src': ['体格检查'], 'regex':'上腹' + self.inner_neg + '肿块', 'default': 0}, #
            {'id':'KZGYT1', 'name': '腹部叩诊鼓音', 'src': ['体格检查', '查体'], 'regex':'(腹部)?(叩诊)?' + self.inner_neg + '鼓音', 'default': 0},
            {'id':'CYKT1', 'name': '肠鸣音亢进（病程早期）', 'src': ['体格检查', '查体'], 'regex':'肠鸣?音((亢进)|(活跃))', 'default': 0},
            {'id':'CYRT1', 'name': '肠鸣音减弱（病程晚期）', 'src': ['体格检查', '查体'], 'regex':'肠鸣?音.?[弱低]', 'default': 0},
            {'id':'HMT1', 'name': '昏迷', 'src': ['体格检查', '查体'], 'regex':'(昏迷)|(意识不清)|(呼之不应)|(意识丧失)|(随意运动消失)|(对外界的刺激的反应迟钝或丧失)|(瞳孔散大)|(对光反射消失)|(双侧瞳孔不等大)|(神经反射消失)', 'default': 0},
            {'id':'TWGT1', 'name': '体温升高', 'src': ['体格检查', '查体', '现病史'], 'regex':'体温' + self.inner_neg + '((3[89])|(4[01])|(37[.][6789]))', 'default': 0},
            {'id':'MBKT1', 'name': '脉搏显著加快', 'src': ['体格检查'], 'regex':'', 'default': 0},
            {'id':'DXYT1', 'name': '低血压', 'src': ['体格检查'], 'regex':'', 'default': 0},
            {'id':'JJZT1', 'name': '反跳痛或肌紧张', 'src': ['体格检查', '现病史', '查体'], 'regex':'(反跳痛)|(肌紧张)|(腹' + self.inner_neg + '紧)|(腹肌抵触感)|(板状腹)|(腹强直)', 'default': 0},
            {'id':'MFST1', 'name': '墨菲征Murphy征', 'src': ['体格检查', '查体'], 'regex':'((墨菲征)|(Murphys?征?：?’?))', 'negregex': '((墨菲征)|(Murphys?))' + self.inner + '阴性', 'default': 0},
            {'id':'FJT1', 'name': '附件肿块', 'src': ['体格检查', '查体', '辅助检查'], 'regex':'(上腹' + self.inner_neg + '肿块)|(附件区?' + self.inner_neg_xx + '[肿包]块)|(附件' + self.inner_neg + '增[粗厚])', 'default': 0},
        ]
        self.feature_id_name_dict = {
            feature["id"]: feature["name"] for feature in self.features
        }

        self.proc_features = None


    def set_proc_features(self, feature_ids):
        """
        设置要提取的特征id列表
        """
        feature_ids = set(feature_ids)
        features = []
        for f in self.features:
            if f['id'] in feature_ids:
                features.append(f)

        self.proc_features = features


    def get_txt_from_records(self, records, src):
        """
        从病理记录中找需要的文字，src为文字路径
        """
        results = []
        for record in records:
            text = ''
            for key in src:
                if key in ['体格检查', '个人史', '年龄', '婚育史', '月经史', '性别', '年龄']:
                    # 此部分通过结构化数据来处理，不做文本处理
                    continue
                elif key == '既往史':
                    text = text + get_json_value(record, ['入院记录', '病史小结', '既往史']).replace('，', '、') + '。'
                    text = text + get_json_value(record, ['入院记录', '既往史']) + '。'
                elif key == '查体':
                    text = text + get_json_value(record, ['首次病程', '病例特点', '查体']) + '。'
                    text = text + get_json_value(record, ['首次病程', '诊断依据']) + '。'
                    text = text + get_json_value(record, ['入院记录', '病史小结', '查体']) + '。'
                    text = text + get_json_value(record, ['入院记录', '专科情况（体检）']) + '。'
                elif key == '辅助检查':
                    text = text + get_json_value(record, ['首次病程', '病例特点', '辅助检查']) + '。'
                    text = text + get_json_value(record, ['入院记录', '病史小结', '辅助检查']) + '。'
                    text = text + get_json_value(record, ['入院记录', '门诊及院外重要辅助检查']) + '。'
                elif key == '病史小结':
                    text = text + get_json_value(record, ['入院记录', '病史小结', '主诉']) + '。'
                    text = text + get_json_value(record, ['入院记录', '病史小结', '既往史']) + '。'
                    text = text + get_json_value(record, ['入院记录', '病史小结', '查体']) + '。'
                    text = text + get_json_value(record, ['入院记录', '病史小结', '辅助检查']) + '。'
                else:
                    text = text + get_json_value(record, [key]) + '。'
                    text = text + get_json_value(record, ['入院记录', key]) + '。'

            results.append(text)

        return results


    def predict_by_model(self, txts, name):
        """
        使用模型预测特征
        """
        # 准备数据
        keywords = [name for i in range(len(txts))]
        labels = ['1' for i in range(len(txts))]

        model = TextClassifier(model_save_path=r'output/models/textclassify',
                                pre_model_path=r"D:/projects/NLP/BertModels/medical-roberta-wwm",
                                num_cls=3,
                                model_name=name,
                                model_file_path=r'D:\projects\NLP\MedicalRecord\FeatureExtraction\data\%s\models\%s.pth' % (self.type, name)
                                )

        model.load_data(txts, labels, label_dict={'0': 0, '1': 1, '2': 2}, texts_pair=keywords, batch_size=8, is_training=False)
        results = model.predict_nowrite()

        results_ = [[r, '', '', txt, 0] for (txt, r) in zip(txts, results)]

        return results_


    def process_common_regex(self, r_strc, match1_strc, txt, record, feature):
        """
        一般正则处理逻辑
        """
        # 正则查找
        regex = feature['regex'] if feature['regex'] != '' else feature['name']
        negregex = feature['negregex'] if 'negregex' in feature else None
        r_regex, match1_regex, match2_regex = self.search_by_regex(txt, regex, negregex, default=(feature['default'] if 'default' in feature else 0))
        match1_regex = '' if match1_regex is None else match1_regex.group(0)
        match2_regex = '' if match2_regex is None else match2_regex.group(0)

        if r_regex != 2:
            return [r_regex, match1_regex, match2_regex, txt, 0]
        elif r_strc == 0:
            return [r_strc, match1_strc, '', txt, 0]
        else:
            return [feature['default'] if 'default' in feature else 0, '', '', txt, 0]


    def process_special_regex_tj(self, r_strc, match1_strc, txt, record, feature):
        """
        处理特殊正则逻辑：停经
        """
        ## bad case : '患者平素月经规则，LMP:2020-01-16，5-7/28-30天，经量中等'
        # regex查找
        # r_regex, match1_regex = self.search_by_regex_simple(txt, feature['regex'], subcls_regex='[。，]')
        r_regex, match1_regex, _ = self.search_by_regex(txt, feature['regex'])
        match1_regex = '' if match1_regex is None else match1_regex.group(0)
        if r_regex == 1:
            return [1, match1_regex, '', txt, 0]
        # 末次月经，月经周期计算
        r_ss, match1_regex = self.search_by_regex_simple(txt, feature['regex_lmp'], subcls_regex='[。，]')
        match1_regex = '' if match1_regex is None else match1_regex.group(0)
        mcyj = self.utils.format_date(match1_regex)
        if mcyj != '' and record['入院时间'] != '':
            r_ss, match2_regex = self.search_by_regex_simple(txt, feature['regex_yjzq'])
            match2_regex = '' if match2_regex is None else match2_regex.group(7)
            yjzq = self.utils.format_num(match2_regex)
            yjzq = 45 if yjzq == '' else int(yjzq) % 100 # 7-8/45 取最后两位
            sub = self.utils.date_sub_date(record['入院时间'], mcyj)
            if sub <= yjzq:
                return [0, match1_regex, match2_regex, txt, 0]
            else:
                return [1, match1_regex, match2_regex, txt, 0]
        else:
            return [r_strc, match1_strc, '', txt, 0]


    def process_special_regex_ercp(self, r_strc, match1_strc, txt, record, feature):
        """
        处理特殊正则逻辑：内镜逆行胰胆管造影、EUS与FNA、主动脉手术胰腺切除术史
        """
        # 手术外伤史
        result1 = self.process_common_regex(r_strc, match1_strc, txt, record, feature)

        # 病史小结，只使用ERCP做正则提取
        feature_ = copy.copy(feature)
        feature_['regex'] = 'ERCP'
        txt = self.get_txt_from_records([record], '病史小结')[0]
        result2 = self.process_common_regex(r_strc, match1_strc, txt, record, feature_)

        #合并结果
        if result1[0] == 1 or result2[0] == 2:
            return result1
        elif result1[0] == 2 or result2[0] == 1:
            return result2
        elif result1[0] == 0 or result1[0] == 3:
            return result1
        else:
            return result2


    def predict_by_regex(self, records, feature):
        """
        使用正则处理特征抽取
        """
        txts = self.get_txt_from_records(records, feature['src'])

        # 是否使用结构化查找
        results = []
        for txt, record in zip(txts, records):
            # 结构化查找
            r_strc, match1_strc = self.search_by_strc(record, feature['name'])

            # 合并结果
            if txt == '' and (r_strc == -1 or r_strc is None):
                results.append([2, '', '', txt, 0])
            elif txt == '' or r_strc == 1:
                results.append([r_strc, match1_strc, '', txt, 0])
            elif feature['name'] == '停经':
                results.append(self.process_special_regex_tj(r_strc, match1_strc, txt, record, feature))
            elif feature['name'] == '内镜逆行胰胆管造影、EUS与FNA、主动脉手术胰腺切除术史':
                results.append(self.process_special_regex_ercp(r_strc, match1_strc, txt, record, feature))
            else:
                results.append(self.process_common_regex(r_strc, match1_strc, txt, record, feature))

        return results


    def search_by_strc(self, record, name):
        """
        通过结构化数据查找
        """
        if '入院记录' not in record:
            return None, ''

        if name == '肿大胆囊':
            text = record['入院记录']['体格检查']['腹部']['胆囊']
            if text.startswith('未触及'):
                return 0, text
            else:
                return 1, text
        elif name == '腹部叩诊鼓音':
            text = record['入院记录']['体格检查']['腹部']['叩诊']
            if '鼓音' in text:
                return 1, text
            else:
                return 0, text
        elif name == '肠鸣音亢进（病程早期）':
            text = record['入院记录']['体格检查']['腹部']['肠鸣音']
            if '亢进' in text:
                return 1, text
            else:
                return 0, text
        elif name == '肠鸣音减弱（病程晚期）':
            text = record['入院记录']['体格检查']['腹部']['肠鸣音']
            if '弱' in text:
                return 1, text
            else:
                return 0, text
        elif name == '肠鸣音消失':
            text = record['入院记录']['体格检查']['腹部']['肠鸣音']
            if '消失' in text:
                return 1, text
            else:
                return 0, text
        elif name == '失血性休克':    #医学生进一步讨论完成
            text = record['入院记录']['体格检查']['腹部']['肠鸣音']
            return 2, text
        elif name == '昏迷':
            text = record['入院记录']['体格检查']['一般情况']['神志']
            if '昏迷' in text:
                return 1, text
            else:
                return 0, text
        elif name == '体温升高':
            text = self.utils.format_temprature(record['入院记录']['体格检查']['生命体征']['体温']).replace('℃', '')
            if text != '':
                if float(text) > 37.5:
                    return 1, text
                else:
                    return 0, text
            else:
                return 2, text
        elif name == '脉搏显著加快':
            text = self.utils.format_num(record['入院记录']['体格检查']['生命体征']['脉搏'])
            if text != '':
                num = int(text)
                if num > 100 and num < 1000:    # 数据错误8979，应该是89，如果>10000，应该是> 100
                    return 1, text
                elif num > 10000:
                    return 1, text
                else:
                    return 0, text
            else:
                return 2, text
        elif name == '低血压':
            text = self.utils.format_pressure(record['入院记录']['体格检查']['生命体征']['血压'])
            arr = text.split('/')
            if len(arr) != 2:
                return 2, text

            ssy = self.utils.format_num(arr[0])
            szy = self.utils.format_num(arr[1])
            if ssy == '' or szy == '':
                return 2, text
            else:
                if int(ssy) < 90 or int(szy) < 60:
                    return 1, text
                else:
                    return 0, text
        elif name == '反跳痛或肌紧张':
            text = record['入院记录']['体格检查']['腹部']['反跳痛']
            if text == '':
                return 2, text
            elif '无' in text:
                return 0, text
            else:
                return 1, text
        elif name == '墨菲征Murphy征':
            text = record['入院记录']['体格检查']['腹部']['Murphy征'].strip()
            if text == '':
                return 2, text
            elif '无' in text or text == '-':
                return 0, text
            else:
                return 1, text
        elif name == '附件肿块':    #医学生进一步讨论完成
            text = record['入院记录']['体格检查']['腹部']['腹部包块'].strip()
            if text.startswith('未触及'):
                return 0, text
            else:
                if '附件' in text:
                    return 1, text
                else:
                    return 0, text
        elif name == '吸烟':
            text = record['入院记录']['个人史']['吸烟史']
            if '无' in text or text.replace('-', '') == '':
                return 0, text
            elif record['入院记录']['个人史']['戒烟时间'] != '':
                return 1, text
            else:
                return 1, text
        elif name == '酗酒':
            text = record['入院记录']['个人史']['饮酒史']
            if '无' in text or text.replace('-', '') == '':
                return 0, text
            else:
                return 1, text
        elif name == '年龄35岁及以上':
            text = record['入院记录']['年龄']
            age = self.utils.format_by_type(text, 'age')
            if age != '':
                if int(age) >= 35:
                    return 1, text
                else:
                    return 0, text
            else:
                return 2, text
        elif name == '既往流产（包括人工流产）':
            num1 = self.utils.format_num(record['入院记录']['婚育史']['自然流产'])
            num2 = self.utils.format_num(record['入院记录']['婚育史']['人工流产'])
            if num1 == '' and num2 == '' and record['入院记录']['婚育史']['产'] == '':
                return 0, '%s，%s' % (num1, num2)
            else:
                num1 = int(num1) if num1 != '' else 0
                num2 = int(num2) if num2 != '' else 0
                if num1 + num2 > 0:
                    return 1, '%s，%s' % (num1, num2)
                else:
                    return 0, '%s，%s' % (num1, num2)
        elif name == '不孕(风险随不孕时间的延长而增加)':
            if '未婚' in record['入院记录']['婚育史']['婚育史']:
                return 0, record['入院记录']['婚育史']['婚育史']
            else:
                has_num = False
                if self.utils.format_num(record['入院记录']['婚育史']['妊娠'], default='0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['产'], default='0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['自然生产'], default='0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['手术生产'], '0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['自然流产'], '0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['人工流产'], '0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['早产'], '0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['死产'], '0') != '0':
                    has_num = True
                if self.utils.format_num(record['入院记录']['婚育史']['引产'], '0') != '0':
                    has_num = True

                if has_num:
                    return 0, str(record['入院记录']['婚育史'])
                else:
                    mar_age = self.utils.format_age(record['入院记录']['婚育史']['结婚年龄'])
                    age = self.utils.format_age(record['入院记录']['年龄'])
                    if mar_age != '' and age != '':
                        # 数据空的认为缺失
                        if int(age) - int(mar_age) > 1 and self.utils.format_num(record['入院记录']['婚育史']['产']) != '':
                            return 1, '结婚年龄：%s，年龄：%s' % (record['入院记录']['婚育史']['结婚年龄'], record['入院记录']['年龄'])
                    return 0, '结婚年龄：%s，年龄：%s' % (record['入院记录']['婚育史']['结婚年龄'], record['入院记录']['年龄'])
        elif name == '停经':
            str_yjs = record['入院记录']['月经史'].strip()
            if str_yjs != '':
                if '是否绝经' in str_yjs:
                    str1, str2 = str_yjs.split('是否绝经')[0], str_yjs.split('是否绝经')[1]
                    if '是' in str2[:3]:
                        return 0, str_yjs
                    elif '末次月经' in str1:
                        str1 = str1.split('末次月经')[1]
                        if '周期' in str1:
                            str1_arr = str1.split('周期')
                            str_mcyj, str_zq = str1_arr[0], str1_arr[1].replace('-', '')
                        else:
                            str_mcyj, str_zq = str1, '45'

                        mcyj = self.utils.format_date(str_mcyj)
                        rysj = record['入院时间']
                        if mcyj != '' and rysj != '':
                            sub = self.utils.date_sub_date(rysj, mcyj)
                            zq = self.utils.format_num(str_zq)
                            zq = 45 if zq == '' else int(zq) % 100 # 周期可能30-45

                            if sub <= zq:
                                return 0, str_yjs
                            else:
                                return 1, str_yjs
            return 2, str_yjs
        elif name == '性别':
            sex = record['入院记录']['性别']
            if sex == '':
                sex = record['入院记录']['病史小结']['性别']
            if sex == '男':
                return 0, sex
            else:
                return 1, sex
        elif name == '年龄':
            age = record['入院记录']['年龄']
            if age == '':
                age = record['入院记录']['病史小结']['年龄']
            if age == '':
                return 38.8, age
            else:
                return float(age), age
        else:
            return None, ''

    # def write_to_excel(self, results, file_path, debug):
    #     print('writing to excel....')
    #     wb = Workbook()
    #     sheet = wb.create_sheet('Sheet1', 0)
    #
    #     span = 4 if debug else 1
    #
    #     # 写表头
    #     for idx, col in enumerate(results.keys()):
    #         if idx < 2:
    #             sheet.cell(1, idx+1).value = col
    #         else:
    #             sheet.cell(1, (idx-2) * span + 3).value = col
    #
    #     # print(list(results.keys()))
    #     # for idx, col in enumerate(results.keys()):
    #     #     print(col, len(results[col]))
    #     #     print(results[col][0])
    #
    #     # 写数据
    #     for idx, col in enumerate(results.keys()):
    #         if idx < 2:
    #             cn = idx + 1
    #             for r_idx, e in enumerate(results[col]):
    #                 sheet.cell(r_idx+2, cn).value = e
    #         else:
    #             cn = (idx-2) * span + 3
    #             for r_idx, e in enumerate(results[col]):
    #                 sheet.cell(r_idx+2, cn).value = e[0]
    #                 if e[4] == 1:
    #                     sheet.cell(r_idx+2, cn).font = self.hl_font1
    #                 elif e[4] == 2:
    #                     sheet.cell(r_idx+2, cn).font = self.hl_font2
    #
    #                 if debug:
    #                     sheet.cell(r_idx+2, cn+1).value = e[1]
    #                     sheet.cell(r_idx+2, cn+2).value = e[2]
    #                     sheet.cell(r_idx+2, cn+3).value = e[3]
    #
    #     wb.save(file_path)
    #     wb.close()


    def process(self, json_file, result_file, labeled_file=None, debug=True):
        """
        处理主程序
        预测结果：{'医保编号': [], '入院日期': [], '特征1': [(结果, 正匹配, 负匹配, 查找文本, 是否和人工标记相同的标志位)]}
        """
        keys, records = load_data(json_file, labeled_file)
        results = {'医保编号': [s.split('_')[0] for s in keys], '入院日期': [s.split('_')[1] for s in keys]}
        for feature in (self.proc_features if self.proc_features else self.features):
            print('processing feature: %s,  type: %s' % (feature['name'], feature['type'] if 'type' in feature else 'regex'))
            # 模型
            if 'type' in feature and feature['type'] == 'model':
                # txts = self.get_txt_from_records(records, feature['src'])
                # f_results = self.predict_by_model(txts, feature['name'])
                # results[feature['id']] = f_results
                results[feature['name']] = ['', '', '', '']
            else:
                f_results = self.predict_by_regex(records, feature)
                # print(feature['name'], f_results)
                # results[feature['id']] = f_results
                results[feature['name']] = f_results


        return results


    def stat_and_tag_results(self, results, manlabeled_data, ignore02=True):
        """
        将正则提取结果和人工标记结果做比对，并统计输出
        输入：
            1. results： "医保编号" -> 特征id -> (result_class, match1, match2, record)
            2. labeled_data："医保编号" -> 特征id -> labeled_class
            3. ignore02：当人工标记和正则提取标志分别为0和2时，是否区分结果是否相同。
        """
        print('state and tag results...')
        mrnos = results['医保编号']
        stat_features = {}
        for key in results.keys():
            if key == '医保编号':
                continue

            fname = key
            for mrno, r in zip(mrnos, results[key]):
                if mrno not in manlabeled_data or fname not in manlabeled_data[mrno]:
                    r[4] = 2
                    continue

                if fname not in stat_features:
                    stat_features[fname] = [0, 0]

                rno, lno = r[0], manlabeled_data[mrno][fname]
                if rno == lno:
                    stat_features[fname][0] = stat_features[fname][0] + 1
                elif ignore02 and ((rno == 0 and lno == 2) \
                    or (rno == 2 and lno == 0)):
                    stat_features[fname][0] = stat_features[fname][0] + 1
                else:
                    # 预测和标记不一样做标志
                    r[4] = 1

        stat_results = [[feature, stat_features[feature][0], stat_features[feature][1]] for feature in stat_features.keys()]
        stat_results = sorted(stat_results, key=lambda x: x[1], reverse=True)
        for line in stat_results:
            print(line)

        return results

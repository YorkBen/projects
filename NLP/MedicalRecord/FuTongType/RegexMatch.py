"""
临床正则提取类
"""
import json
import re
import logging
import copy
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, Border, Side, PatternFill, colors, Alignment
import sys
import math
import argparse

sys.path.append('../Lib')
from RegexBase import RegexBase
from RegexUtil import RegexUtil
from MRRecordUtil import *


class RegexMatch(RegexBase):
    def __init__(self):
        super(RegexMatch, self).__init__()

        self.utils = RegexUtil()
        self.hl_font1 = Font(name="Arial", size=14, color="00FF0000")
        self.hl_font2 = Font(name="Arial", size=14, color="0000FF00")

        # 疾病对应的正则
        self.features = [
            {'name': '缓解', 'regex': '(缓解)|(好转)|(改善)|(减轻)', 'exclude_prefix': '((改变体位)|(被动体位)|(体位变动)|(变换体位))|((平卧)|(卧位)|(平躺))|((直立位)|(站立[时后]?))|(走动)|((弯曲位)|(弯腰[时后]?))|((前倾位)|(前倾[时后]?))|(不愿意动)|(婴儿姿势蜷卧)|(蹲下)', 'match_type': ['内脏痛'], 'neg_match_type': ['躯体痛'], 'neg_score': 1},
            # {'name': '加剧', 'regex': '(加剧)|(加重)', 'exclude_prefix': '进食后', 'match_type': ['躯体痛']},
            {'name': '反跳痛', 'regex':'反跳痛', 'match_type': ['躯体痛'], 'neg_match_type': ['内脏痛'], 'score': 10},
            {'name': '肌紧张', 'regex':'(肌紧张)|(板状腹)|(腹' + self.inner_neg + '紧)|(腹部变硬)|(无意识保护)|([据拒]按)', 'match_type': ['躯体痛'], 'neg_match_type': ['内脏痛'], 'score': 10},
            {'name': '肌紧张', 'regex':'(腹肌抵触感)|(腹' + self.inner_neg + '强直)|(肌卫)', 'match_type': ['躯体痛'], 'neg_match_type': ['内脏痛']},
            {'name': '肠鸣音亢进', 'regex':'肠鸣?音((亢进)|(活跃))', 'match_type': ['内脏痛']},
            {'name': '肠鸣音减弱', 'regex':'肠鸣?音.?(([弱低])|(消失))', 'match_type': ['躯体痛'], 'score': 2},
            {'name': '疼痛叙述不清', 'regex':'疼痛' + self.inner_neg + '叙述不清', 'match_type': ['内脏痛']},
            {'name': '饭前痛', 'regex':'(饭|(进食))前' + self.inner_neg + '痛', 'match_type': ['内脏痛'], 'score': 2},
            {'name': '进食后', 'regex':'进食后' + self.inner_neg + '((好转)|(加重)|(减轻)|(变化))', 'match_type': ['内脏痛'], 'neg_match_type': ['躯体痛']},
            {'name': '排便后', 'regex':'(排便后' + self.inner_neg + '((好转)|(出现)|(减轻)))|(大便后感疼痛)', 'match_type': ['内脏痛']},
            # {'name': '运动后', 'regex':'((运动)|(活动))后' + self.inner_neg + '((好转)|(减轻)|(加重)|(变化))', 'match_type': ['内脏痛']},
            {'name': '强迫体位', 'regex':'((改变体位)|(被动体位)|(体位变动)|(变换体位))|((平卧)|(卧位)|(平躺))|((直立位)|(站立[时后]?))|(走动)|((弯曲位)|(弯腰[时后]?))|((前倾位)|(前倾[时后]?))|(不愿意动)|(婴儿姿势蜷卧)|(蹲下)', 'match_type': ['躯体痛'], 'score': 10},
            {'name': '体位无关', 'regex':'(与体位无关)|(不随体位改变)', 'match_type': ['内脏痛']},
            {'name': '经常性体位改变', 'regex':'经常性体位改变', 'match_type': ['内脏痛']},
            {'name': '压痛', 'regex':'压痛', 'match_type': ['内脏痛', '躯体痛', '牵涉痛']},
            {'name': '深压痛', 'regex':'((深|轻)' + self.inner_neg + '压痛)|(压痛' + self.inner_neg + '(深|轻))', 'match_type': ['内脏痛']},
            {'name': '压痛明显', 'regex':'(压痛' + self.inner_neg + '明显)|(明显' + self.inner_neg + '压痛)', 'match_type': ['躯体痛'], 'score': 10},
            {'name': '胃肠型', 'regex':'(可见' + self.inner_neg_xx + '胃肠型)|(胃肠型([（(]?[+][）)]?))', 'match_type': ['内脏痛']},
            {'name': '蠕动波', 'regex':'(可见' + self.inner_neg_xx + '蠕动波)|(蠕动波([（(]?[+][）)]?))', 'match_type': ['内脏痛']},
            {'name': '腹式呼吸', 'regex':'腹式呼吸((-)|(消失))', 'match_type': ['躯体痛'], 'score': 10},
            {'name': '腹膨隆', 'regex':'腹' + self.inner_neg + '膨隆', 'match_type': ['躯体痛']},
            {'name': '肝浊音界', 'regex':'肝浊音界消失', 'match_type': ['躯体痛'], 'score': 10},
            {'name': '墨菲征', 'regex':'((墨菲氏?征)|(Murphy[\'’]?s?征?[：:]?))(([（(]?[+][）)]?)|(阳性)|(可疑))', 'match_type': ['躯体痛'], 'score': 10},
            {'name': '突发', 'regex':'(突然发作)|(突感)|(小时前)|(突发)', 'match_type': ['躯体痛', '牵涉痛']},

            {'name': '程度一般', 'regex':'(不适)|(程度一般)|(可耐受)|(未在意)|(不剧烈?)|(轻度)', 'match_type': ['内脏痛']},
            {'name': '间断发作', 'regex':'(间断性)|(间歇性)|(阵发性)|(阵痛)|(阵发加重)|(反复发作)|(偶有)|(病情易反复)|(迁延不愈)|(一过性)', 'match_type': ['内脏痛']},
            {'name': '固定体位', 'regex':'([左右中]{1,2}[上下]腹)|([左右]侧)|(剑突下)|(附件区)', 'match_type': ['躯体痛', '牵涉痛']},
            {'name': '固定体位', 'regex':'(麦氏点)|(Mc Bunery 点)|(部位较?固定)|(固定于?)|(局限于?)|(集中于?)', 'match_type': ['躯体痛', '牵涉痛'], 'score': 10},
            {'name': '定位模糊', 'regex':'([上下]腹)|(([左右]侧)?腰腹部?)|(腰痛)|(([左右]侧)?腰背部?)|(双侧)|(位置不固定)|(游走性)|(定位不清)|(部位不确定)|(部位不明确)|(对称性)|(散在)', 'match_type': ['内脏痛']},
            {'name': '定位模糊', 'regex':'(全腹)', 'exclude_postfix': '无|(未见)|(平软)', 'match_type': ['内脏痛']},
            {'name': '定位模糊', 'regex':'(脐周)', 'match_type': ['内脏痛'], 'score': 5},
            {'name': '牵扯', 'regex':'((牵扯)|(痉挛))[样性]', 'match_type': ['内脏痛']},
            {'name': '坠胀', 'regex':'(摇举痛)|(绞痛)|(坠胀)|(深部)', 'match_type': ['内脏痛']},
            {'name': '消化道症状', 'regex':'(恶心)|(呕吐)|(食欲减退)|(腹胀[^痛])|(反酸)|(嗳气)|(打嗝)|(干呕)|(烧心)|(便血)', 'match_type': ['内脏痛']},
            {'name': '排便', 'regex':'(干便)|(便秘)', 'match_type': ['内脏痛']},
            {'name': '排便', 'regex':'((未排)|(未解)|(停止))((排气)|(排便)|(大便))', 'match_type': ['内脏痛']},
            {'name': '排便', 'regex':'((排气)|(排便)|(大便))((未解)|(减少)|(干结)|(未行)|(排出不畅)|(困难))', 'match_type': ['内脏痛']},

            {'name': '大便', 'regex':'(稀便)|(清水' + self.inner_neg + '便)|(黑便)|(糊状大便)|(便秘与腹泻交替)|(伴解大便感)|(里急后重)|(大便次数增多)|(大便[不无]规[律则])', 'match_type': ['内脏痛']},
            {'name': '夜间发作', 'regex':'(夜间发作)|(凌晨发作)|(多发于凌晨)', 'match_type': ['内脏痛']},
            {'name': '慢性腹痛', 'regex':'(慢性' + self.inner_neg + '腹痛)|(腹痛' + self.inner_neg + '慢性)', 'match_type': ['内脏痛']},
            {'name': '腹胀程度逐渐加重', 'regex':'(腹胀' + self.inner_neg + '加重)', 'match_type': ['内脏痛']},

            {'name': '钝痛', 'regex':'钝痛', 'match_type': ['内脏痛']},
            {'name': '锐痛', 'regex':'(刀割样)|(针扎样)|(撕裂样)|(刺痛)', 'match_type': ['躯体痛']},
            {'name': '程度剧烈', 'regex':'(激烈)|(剧烈)|(不可忍受)|(难以忍受)|(性质较剧)', 'match_type': ['躯体痛']},
            {'name': '持续', 'regex':'(持续性)|(痛' + self.inner_neg + '持续)', 'match_type': ['躯体痛']},
            {'name': '皮疹', 'regex':'(皮疹)|(带状斑疹)|(出血点)', 'match_type': ['躯体痛']},
            {'name': '转移', 'regex':'(转移至全腹)|(随后全腹)|(多处转移)', 'match_type': ['躯体痛']},
            {'name': '转移', 'regex':'(扩布至全腹)|(疼痛向上延续)|(蔓延至)|(延伸至)', 'match_type': ['躯体痛'], 'score': 10},
            {'name': '动作后改变', 'regex':'(深呼吸[时后]?)|(大笑[时后]?)|(咳嗽[时后]?)', 'match_type': ['躯体痛']},
            {'name': '面容痛苦', 'regex':'(急性' + self.inner_neg + '面容)|(痛苦' + self.inner_neg + '面容)|(面容痛苦)|(呼吸急促)|(不愿说话)|(出汗)|(冷汗)|(大汗)', 'match_type': ['躯体痛']},
            {'name': '牵涉', 'regex':'(放射痛?)|(辐射)|(牵涉)|(反射至)|((伴|(偶有))腰背部?不适)|(肩背部)|(伴后?背部?[疼酸胀剧]?痛)', 'match_type': ['牵涉痛'], 'score': 100},
            # {'name': '牵涉', 'regex':'(放射痛)|(((转移)|(辐射)|(牵涉)|(放射)|(可波及)|(伴))' + self.inner_neg + '[腰肩腿肢臂])|([腰背肩腿肢臂]' + self.inner_neg + '((转移)|(辐射)|(牵涉)|(放射)|(可波及)|(伴)))', 'match_type': ['牵涉痛']},
            {'name': '牵涉', 'regex':'((转移)|(伴腰腹?痛)|([及伴]腰部酸胀)|(伴双侧腰胀)|(可波及左侧腰部)|(随后出现持续性背痛不适))|(((转移)|(可波及)|(伴))' + self.inner_neg + '[腰肩腿臂])', 'match_type': ['牵涉痛'], 'score': 10},
            {'name': '神经性', 'regex': '(活动' + self.inner_neg + '受限)|(感觉过敏)|(感觉丧失)', 'match_type':['神经性疼痛'], 'score': 10},
            {'name': '神经性', 'regex': '不能久[坐站]|(对?侧' + self.inner_neg + '皮疹)', 'match_type':['神经性疼痛'], 'score': 2},
        ]


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

    def preprocess_txt1(self, text):
        """
        删除句子中无关信息
        """
        arr_0 = text.split('。')
        r_0 = []
        for t0 in arr_0:
            arr_1 = t0.split('，')
            r_1 = []
            skip = False
            for s in arr_1:
                logging.debug(s)
                appended = False

                if re.search('([B彩超声]{2}.*[;：:])|([起发]病.*来)|(理结果回报[：:])|(病检.*((示)|(结果))[：:])|(诊断为：)|(((血常规)|(尿常规)|(MRI)|(MRCP)|(CT)|(成像)|(检测)|(测定)|([三四五六七八九]项)).*[：:])|(腹部.*：)', s):
                    logging.debug('skip rest!')
                    skip = True

                if re.search('(血HCG)|(考虑)|(可能)|(细胞)|(酶)|(蛋白)|(无明显好转)|(转氨酶)|(进一步)|(具体不详)|(((腹部平片)|(检查)|(彩超)).*示)|(诊断为)|(平扫)', s):
                    logging.debug('skip current!')
                    continue

                if re.search('([^镇止]痛)|(缓解)', s):
                    r_1.append(s)
                    logging.debug('append!')
                    appended = True

                if re.search('(为求诊治)|([于在].*[治疗就诊处理]{2})|([急门]诊.*(以拟))|(以.*[收入住院]{2})|(我[院科].*诊)|([行予].*((治疗)|(处理)|术))|(皮肤)|(口服)|(服用)|(缓解)|(就诊)|(MRI)|(MRCP)|(CT)|(成像)', s):
                    logging.debug('skip current!')
                    continue

                if not skip and not appended:
                    logging.debug('not skip, add!')
                    r_1.append(s)

            if len(r_1) > 0:
                r_0.append('，'.join(r_1))
        s_0 = '。'.join(r_0)
        return s_0.strip()

    def get_txt_from_record(self, record):
        """
        从病理记录中找需要的文字，src为文字路径
        """
        text = ''
        # text = text + get_json_value(record, ['入院记录', '病史小结', '主诉'])
        # text = text + '，' if len(text) > 0 and text[-1] not in ['。', '，'] else text
        text = text + self.preprocess_txt1(get_json_value(record, ['入院记录', '现病史']))
        text = text + '，' if len(text) > 0 and text[-1] not in ['。', '，'] else text
        text = text + get_json_value(record, ['入院记录', '专科情况（体检）'])
        text = text.strip().replace(',', '，')

        return text


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


    def process_regex(self, text):
        """
        处理主程序
        预测结果：
        """
        texts = re.split('[，。；]', text)
        result = []
        t_start = 0
        for t in texts:
            logging.debug(t)
            for feature in self.features:
                logging.debug(feature['regex'])
                for match in re.finditer(feature['regex'], t, re.I):
                # match = re.search(feature['regex'], t, re.I)
                    if match is not None:
                        word = match.group(0)
                        # 前缀or后缀排除
                        if 'exclude_prefix' in feature or 'exclude_postfix' in feature:
                            try:
                                if 'exclude_prefix' in feature:
                                    pmatch, _ = self.check_prefix(t, word, feature['exclude_prefix'])
                                    if pmatch:
                                        break
                                elif 'exclude_postfix' in feature:
                                    pmatch, _ = self.check_postfix(t, word, feature['exclude_postfix'])
                                    if pmatch:
                                        break
                            except:
                                print('Except: word not in text!')

                        # 否定词
                        nmatch, _ = self.check_neg_word_findpos(t, word)

                        # 写结果
                        start, end = match.span()
                        score, neg_score = feature['score'] if 'score' in feature else 1, feature['neg_score'] if 'neg_score' in feature else 1
                        if nmatch and 'neg_match_type' in feature:
                            result.append((feature['neg_match_type'], start + t_start, end + t_start, word, nmatch, neg_score))
                        elif not nmatch:
                            result.append((feature['match_type'], start + t_start, end + t_start, word, nmatch, score))

            t_start = t_start + len(t)

        result = sorted(result, key=lambda x: (x[1], -x[2])) #先按开始位置谁靠前，再按结束位置谁靠后

        return result

    def merge_words(self, arr):
        def check_intersect(elem_1, elem_2):
            pos1 = [elem_1[1], elem_1[2]]
            pos2 = [elem_2[1], elem_2[2]]
            s1 = {i for i in range(pos1[0], pos1[1])}
            s2 = {i for i in range(pos2[0], pos2[1])}
            s = s1.intersection(s2)
            return len(s) > 0

        def check_same_word(elem_1, elem_2):
            return elem_1[3] == elem_2[3]

        def one_iter(arr, func):
            if len(arr) == 0:
                return []

            results, idx = [arr[0]], 1
            while idx < len(arr):
                elem_1, elem_2 = results[-1], arr[idx]
                if func(elem_1, elem_2):
                    pass
                else:
                    results.append(elem_2)
                idx += 1

            return results

        results = one_iter(arr, check_intersect)
        results = one_iter(results, check_same_word)

        return results

    def split_text_by_time_span(self, text):
        # time_re = '(([0-9]{2,4}[-/年.])?[0-9]{1,2}[-/月.][0-9]{1,2})|([0-9]{2,4}[-/年.][0-9]{1,2}[-/月.]([0-9]{1,2})?)|([0-9]{1,2}[:：]([0-9]{2}))'
        time_re = '((([12][09])?[0-9]{2}[-/年.])?[01]?[0-9][-/月.][0-3]?[0-9][^0-9%*.A-Za-z\/\^\-℃次])|(([12][09])?[0-9]{2}[-/年.][01]?[0-9][^0-9%*.A-Za-z\/\^\-℃次])|([0-2]?[0-9][:：][0-6]?[0-9][^0-9%*.A-Za-z\/\^\-℃次])|([一二三四五六七八九十两0-9][点时分号岁天月日周]([^\/次每]|$))'
        time_re_neg_prefix = '(持续)|(月经)|(经期)|(周期)|(\-)|(\/)'
        # re1 = '[今凌前当昨半上中下 年月日周午旬  早晚夜晨天 点时分 时间数余小号个一二三四五六七八九十0-9+]{2,}'
        # re1_n = '[一二三四五六七八九十0-9+]'
        re1 = '([今前昨当凌][日早晚天夜晨])|([上中下本半][午年月日周旬天])|(近期)'

        re2 = '(近来)|([一数]个?[月日年][来前]?)|(傍晚)|(去年)|([^出表呈发]现)|今|(翌日)|(长期)|(次日)|(晨起)|(既往[^史月经])|((([早午晚][饭餐]))[前后])|(小时' + self.inner_neg + '前)'
        re3 = '[0-9:-]((AM)|(am)|h|(PM)|(pm))'
        re4 = '于.*?左右'
        re5 = '(后[^背腰方肩])|(很快)|(再次)|(再发)'
        re5_n = '(术后)|(病后)|(治疗后)|(里急后重)|(处理后)|(服' + self.inner_neg_xx + '后)' #|(休息后)|(活动后)|(饮水后)|(进食后)

        def check_time_str(text):
            text = re.sub('[(（].*[）)]', '', text)
            for match in re.finditer(time_re, text):
                if match is not None:
                    word = match.group(0)
                    if not self.check_prefix(text, word, time_re_neg_prefix)[0]:
                        return True

            for regex in [re1, re2, re3, re4]:
                for match in re.finditer(regex, text):
                    if match is not None:
                        return True

            for match in re.finditer(re5, text):
                if match is not None and not re.search(re5_n, text):
                    return True

            return False

        texts = re.split('[，。；]', text)
        result, new_span = [], []
        for text in texts:
            if check_time_str(text):
                if len(new_span) > 0:
                    result.append(new_span)
                new_span = [text]
            else:
                new_span.append(text)
        result.append(new_span)
        return result

    def process_by_spans(self, text_spans):
        result, log_types = [], []
        pre_types = []
        for texts in text_spans:
            text = '，'.join(texts)
            pre_types.extend(self.merge_words(self.process_regex(text)))
            r = self.filt_max(pre_types)
            if r is not None:
                if len(result) == 0 or result[-1] != r:
                    result.append(r)
                    log_types.append(pre_types)
                else:
                    log_types[-1].extend(pre_types)
                pre_types = []

        if len(result) == 0:
            r = self.filt_max(pre_types, delta=1)
            if r is not None:
                if len(result) == 0 or result[-1] != r:
                    result.append(r)
                    log_types.append(pre_types)
                else:
                    log_types[-1].extend(pre_types)


        return result, log_types


    def filt_max(self, arr, delta=3):
        """
        每个时间span中选取最大分数的类别，并根据delta值过滤
        """
        if len(arr) == 0:
            return None

        type_score = {}
        for item in arr:
            item_score = item[-1]
            for type in item[0]:
                if type not in type_score:
                    type_score[type] = 0
                type_score[type] += item_score / len(item[0])

        type_score_arr = sorted([(k, v) for k,v in type_score.items()], key=lambda x: x[1], reverse=True)
        if type_score_arr[0][1] >= delta:
            return type_score_arr[0][0]
        else:
            return None


    def db_scan(self, arr):
        def max_density(sub_arr, score_arr):
            type_word_set, type_dict = {}, {}
            for e_arr, score in zip(sub_arr, score_arr):
                for e in e_arr:
                    if e not in type_dict:
                        type_dict[e] = 0
                    # if '%s_%s' % (e, word) not in type_word_set:
                    #     type_dict[e] = type_dict[e] +1
                    #     type_word_set.add('%s_%s' % (e, word))
                    type_dict[e] = type_dict[e] + score / len(e_arr)
            r = sorted([(k, v) for k, v in type_dict.items()], key=lambda x: x[1], reverse=True)
            if len(r) > 0:
                return r[0]
            else:
                return None, 0

        def unify_arr_label(sub_arr, label):
            result = []
            for idx in range(len(sub_arr)):
                if label in sub_arr[idx]:
                    result.append([label])
                else:
                    result.append(sub_arr[idx])

            return result

        types_arr = [x for x, _, _, w, _, s in arr]
        words_arr = [w for x, _, _, w, _, s in arr]
        score_arr = [s for x, _, _, w, _, s in arr]

        radius, step, delta = 2, 1, 3
        window_size = radius * 2 + 1
        # end_idx = window_size
        # sub_arr = types_arr[:min(end_idx, len(types_arr))]
        # rtype, score = max_density(sub_arr, score_arr[:min(end_idx, len(types_arr))])
        # result = [rtype] if score >= delta else []
        # print(rtype, score)
        # if score >= delta:
        #     types_arr[:min(end_idx, len(types_arr))] = unify_arr_label(sub_arr, rtype)
        # while end_idx < len(types_arr):
        #     end_idx = end_idx + step
        #     sub_arr = types_arr[end_idx-window_size:end_idx]
        #     rtype, score = max_density(sub_arr, score_arr[end_idx-window_size:end_idx])
        #     print(rtype, score)
        #     if score >= delta:
        #         types_arr[end_idx-window_size:end_idx] = unify_arr_label(sub_arr, rtype)
        #         if len(result) == 0 or result[-1] != rtype:
        #             result.append(rtype)
        cur_idx, arr_len = 0, len(types_arr)
        result = []
        while cur_idx <= arr_len - 1:
            start, end = max(0, cur_idx - radius), min(arr_len, cur_idx + radius + 1)
            rtype, score = max_density(types_arr[start:end], score_arr[start:end])
            if score >= delta or len(types_arr) <= radius + 1:
                # types_arr[start:end] = unify_arr_label(types_arr[start:end], rtype)
                if len(result) == 0 or result[-1] != rtype:
                    result.append(rtype)

            cur_idx = cur_idx + 1
            while len(types_arr) > start + 1 and types_arr[start + 1] == [rtype]:
                cur_idx = cur_idx + 1
                start = start + 1

        return result


def merge_match_labels(arr):
    result = []
    for e in arr:
        if (len(result) == 0 or result[-1] != e) and e != '未知':
            result.append(e)
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        help=
        '输入文件路径')
    parser.add_argument(
        '--idx',
        type=int,
        help=
        '输入文件路径')
    parser.add_argument(
        '--end_idx',
        type=int,
        help=
        '输入文件路径')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Whether Split data for training development')


    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=1)

    with open('合并后数据32.json', encoding="utf-8") as f:
        json_data = json.load(f, strict=False)

    rm = RegexMatch()

    not_same_ct = 0
    with open('result.txt', 'w') as f:
        for idx, item in enumerate(json_data):
            if args.idx:
                if args.idx != idx:
                    continue

            text = item['txt']
            # print(text)
            # text = rm.get_txt_from_record(record)
            # print(text)


            # arr = rm.merge_words(rm.process_regex(text))
            # result = rm.db_scan(arr)
            text_arr = rm.split_text_by_time_span(text)
            result, log_types =  rm.process_by_spans(text_arr)

            golden = merge_match_labels([e['label'] for e in item['match_list']])

            if args.end_idx and idx <= args.end_idx:
                print(idx)
                print(text)
                # print(item['match_list'])
                # print(golden)
                # print(result)
                print(text_arr)
                # print(log_types)
                print('----------------------------------')

            if result == golden:
                for match_item in item['match_list']:
                    f.write('%s\t%s\t%s\t一致\n' % (item['txt'], match_item["match_str"], match_item["label"]))
            else:
                not_same_ct = not_same_ct + 1
                for match_item in item['match_list']:
                    f.write('%s\t%s\t%s\t不一致\t%s\t%s\n' % (item['txt'], match_item["match_str"], match_item["label"], log_types, result))

    print((len(json_data) - not_same_ct) / len(json_data), (len(json_data) - not_same_ct), len(json_data))

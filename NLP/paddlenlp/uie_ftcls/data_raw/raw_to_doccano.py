import json
import argparse
import random
import re
import logging

random.seed(1234)

def preprocess_txt1(text):
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

def preprocess_txt2(text):
    """
    处理体格检查
    """
    arr_0 = text.split('。')
    r_0 = []
    for t0 in arr_0:
        arr_1 = t0.split('，')
        r_1 = []
        for s in arr_1:
            if re.search('(腹)|(反跳痛)|(肌紧张)|(压痛)|(墨菲氏征)|(Murphy)|(麦氏点)', s):
                r_1.append(s)

        if len(r_1) > 0:
            r_0.append('，'.join(r_1))
    s_0 = '，'.join(r_0)

    return s_0.strip()

inner_neg = '[^，；、。无未不非]{,4}'
inner_neg_xx = '[^，；、。无未不非]{,20}'
features = [
    {'name': '缓解', 'regex': '(缓解)|(好转)|(改善)', 'match_type': ['内脏痛'], 'neg_match_type': ['躯体痛']},
    {'name': '反跳痛', 'regex':'反跳痛', 'match_type': ['躯体痛'], 'neg_match_type': ['内脏痛'], 'score': 10},
    {'name': '肌紧张', 'regex':'(肌紧张)|(板状腹)|(腹' + inner_neg + '紧)|(腹部变硬)|(无意识保护)|([据拒]按)', 'match_type': ['躯体痛'], 'neg_match_type': ['内脏痛'], 'score': 10},
    {'name': '肌紧张', 'regex':'(腹肌抵触感)|(腹' + inner_neg + '强直)|(肌卫)', 'match_type': ['躯体痛'], 'neg_match_type': ['内脏痛']},
    {'name': '肠鸣音亢进', 'regex':'肠鸣?音', 'match_type': ['内脏痛']},
    {'name': '疼痛叙述不清', 'regex':'疼痛' + inner_neg + '叙述不清', 'match_type': ['内脏痛']},
    {'name': '进食后', 'regex':'进食后' + inner_neg + '((加重)|(减轻)|(变化))', 'match_type': ['内脏痛'], 'neg_match_type': ['躯体痛']},
    {'name': '排便后', 'regex':'(排便后' + inner_neg + '((好转)|(出现)|(减轻)))|(大便后感疼痛)', 'match_type': ['内脏痛']},
    {'name': '运动后', 'regex':'运动后' + inner_neg + '((好转)|(减轻))', 'match_type': ['内脏痛']},
    {'name': '强迫体位', 'regex':'((改变体位)|(体位变动)|(变换体位))|((平卧)|(卧位)|(平躺))|((直立位)|(站立[时后]?))|(走动)|((弯曲位)|(弯腰[时后]?))|((前倾位)|(前倾[时后]?))|(不愿意动)|(婴儿姿势蜷卧)|(蹲下)', 'match_type': ['躯体痛'], 'score': 10},
    {'name': '体位无关', 'regex':'(与体位无关)|(不随体位改变)', 'match_type': ['内脏痛']},
    {'name': '经常性体位改变', 'regex':'经常性体位改变', 'match_type': ['内脏痛']},
    {'name': '压痛', 'regex':'压痛', 'match_type': ['内脏痛', '躯体痛', '牵涉痛']},
    {'name': '深压痛', 'regex':'((深|轻)' + inner_neg + '压痛)|(压痛' + inner_neg + '(深|轻))', 'match_type': ['内脏痛']},
    {'name': '压痛明显', 'regex':'(压痛' + inner_neg + '明显)|(明显' + inner_neg + '压痛)', 'match_type': ['躯体痛'], 'score': 10},
    {'name': '胃肠型', 'regex':'(可见' + inner_neg_xx + '胃肠型)|(胃肠型([（(]?[+][）)]?))', 'match_type': ['内脏痛']},
    {'name': '蠕动波', 'regex':'(可见' + inner_neg_xx + '蠕动波)|(蠕动波([（(]?[+][）)]?))', 'match_type': ['内脏痛']},
    {'name': '腹式呼吸', 'regex':'腹式呼吸', 'match_type': ['躯体痛'], 'score': 10},
    {'name': '腹膨隆', 'regex':'腹' + inner_neg + '膨隆', 'match_type': ['躯体痛']},
    {'name': '肝浊音界', 'regex':'肝浊音界消失', 'match_type': ['躯体痛'], 'score': 10},
    {'name': '墨菲征', 'regex':'(墨菲氏?征)|(Murphy[\'’]?s?征?[：:]?)', 'match_type': ['躯体痛'], 'score': 10},
    {'name': '突发', 'regex':'(突然发作)|(突感)|(小时前)|(突发)', 'match_type': ['躯体痛', '牵涉痛']},

    {'name': '程度一般', 'regex':'(不适)|(程度一般)|(可耐受)|(未在意)|(不剧烈)', 'match_type': ['内脏痛']},
    {'name': '间断发作', 'regex':'(间断性)|(间歇性)|(阵发性)|(阵痛)|(阵发加重)|(反复发作)|(偶有)|(病情易反复)|(迁延不愈)|(一过性)', 'match_type': ['内脏痛']},
    {'name': '固定体位', 'regex':'([左右]中?[上下]腹)|([左右]侧)|(剑突下)|(附件区)', 'match_type': ['躯体痛', '牵涉痛']},
    {'name': '固定体位', 'regex':'(麦氏点)|(Mc Bunery 点)|(部位较?固定)|(固定于?)|(局限于?)', 'match_type': ['躯体痛', '牵涉痛'], 'score': 10},
    {'name': '定位模糊', 'regex':'(脐周)|([上下]腹)|(腰腹部)|(腰痛)|(全腹)|(腰背部)|(双侧)|(位置不固定)|(游走性)|(定位不清)|(部位不确定)|(对称性)|(散在)', 'match_type': ['内脏痛']},
    {'name': '牵扯', 'regex':'((牵扯)|(痉挛))[样性]', 'match_type': ['内脏痛']},
    {'name': '坠胀', 'regex':'(摇举痛)|(绞痛)|(坠胀)|(深部)', 'match_type': ['内脏痛']},
    {'name': '消化道症状', 'regex':'(恶心)|(呕吐)|(食欲减退)|(腹胀[^痛])|(反酸)|(嗳气)|(打嗝)|(干呕)|(烧心)|(便血)', 'match_type': ['内脏痛']},
    {'name': '排便', 'regex':'(干便)|(便秘)', 'match_type': ['内脏痛']},
    {'name': '排便', 'regex':'(未排)|(未解)|(停止)|(排气)|(排便)|(大便)|(减少)|(干结)|(未行)|(排出不畅)|(困难)|(量少)', 'match_type': ['内脏痛']},

    {'name': '大便', 'regex':'(稀便)|(清水' + inner_neg + '便)|(黑便)|(糊状大便)|(便秘与腹泻交替)|(伴解大便感)|(里急后重)|(大便次数增多)', 'match_type': ['内脏痛']},

    {'name': '钝痛', 'regex':'钝痛', 'match_type': ['内脏痛']},
    {'name': '锐痛', 'regex':'(刀割样)|(针扎样)|(撕裂样)|(刺痛)', 'match_type': ['躯体痛']},
    {'name': '程度剧烈', 'regex':'(激烈)|(剧烈)|(不可忍受)|(难以忍受)|(性质较剧)', 'match_type': ['躯体痛']},
    {'name': '锐痛', 'regex':'(刀割样)|(针扎样)|(撕裂样)|(刺痛)', 'match_type': ['躯体痛']},
    {'name': '持续', 'regex':'(持续性)|(痛' + inner_neg + '持续)', 'match_type': ['躯体痛']},
    {'name': '皮疹', 'regex':'(皮疹)|(带状斑疹)', 'match_type': ['躯体痛']},
    {'name': '转移', 'regex':'(转移至全腹)|(随后全腹)|(多处转移)', 'match_type': ['躯体痛']},
    {'name': '转移', 'regex':'(扩布至全腹)|(疼痛向上延续)|(蔓延至)|(延伸至)', 'match_type': ['躯体痛'], 'score': 10},
    {'name': '动作后改变', 'regex':'(深呼吸[时后]?)|(大笑[时后]?)|(咳嗽[时后]?)', 'match_type': ['躯体痛']},
    {'name': '面容痛苦', 'regex':'(急性' + inner_neg + '面容)|(痛苦' + inner_neg + '面容)|(面容痛苦)|(呼吸急促)|(不愿说话)', 'match_type': ['躯体痛']},
    {'name': '牵涉', 'regex':'(放射痛)|(放射)|(辐射)|(牵涉)|(伴腰背部不适)|(肩背部)|(伴后背痛)|(伴后背部疼痛)', 'match_type': ['牵涉痛'], 'score': 100},
    # {'name': '牵涉', 'regex':'(放射痛)|(((转移)|(辐射)|(牵涉)|(放射)|(可波及)|(伴))' + inner_neg + '[腰肩腿肢臂])|([腰背肩腿肢臂]' + inner_neg + '((转移)|(辐射)|(牵涉)|(放射)|(可波及)|(伴)))', 'match_type': ['牵涉痛']},
    {'name': '牵涉', 'regex':'((转移)|(伴腰痛)|(伴腰腹痛)|(伴双侧腰胀)|(可波及左侧腰部))|(((转移)|(可波及)|(伴))' + inner_neg + '[腰肩腿肢臂])', 'match_type': ['牵涉痛'], 'score': 10},
]

time_re = '(([0-9]{2,4}[-/年.])?[0-9]{1,2}[-/月.][0-9]{1,2})|([0-9]{2,4}[-/年.][0-9]{1,2}[-/月.]([0-9]{1,2})?)|([0-9]{1,2}[:：]([0-9]{2}))'
re1 = '[今凌前当昨半上中下年月日周时分旬早晚晨午夜天点时间数余近期小两号个岁一二三四五六七八九十0-9+]{2,}'
re1_n = '[一二三四五六七八九十0-9+]'
re2 = '(近来)|(傍晚)|(小时)|(去年)|([^出]现)|今|(翌日)|(长期)|(次日)|(晨起)|(既往)|((([早午晚][饭餐])|(进食.*))[前])'
re3 = '[0-9:-]((AM)|(am)|h|(PM)|(pm))'
re4 = '于.*?左右'
re5 = '(后)|(很快)|(再次)|(再发)'

def process_text_regex(text):
    results = []
    texts = re.split('[，。；]', text)
    for t in texts:
        t = t.strip()
        added = False
        if len(t) == 0:
            continue

        if re.search('(血HCG)|(考虑)|(可能)|(细胞)|(回声)|(酶)|(蛋白)|(脉搏)|(呼吸)|(体温)|(转氨酶)|(进一步)|(具体不详)|(((腹部平片)|(检查)|(彩超)|(超声)).*示)|(诊断为)|(平扫)|(附件区' + inner_neg + '(影|囊|灶|(包块)|(增厚)))', t):
            continue

        if re.search('([B彩超声]{2}.*[;：:])|([起发]病.*来)|(病理)|(病检)|(镜下)|(肠镜)|(内镜)|(退镜)|(活检)|(免疫组化)|(腺体)|(DNA)|(理结果回报[：:])|(为求诊?治)|([于在].*[治疗就诊处理]{2})|([急门]诊.*(以拟))|(以.*[收入住院]{2})|(我[院科].*诊)|(查血)|(瘢痕)|(切口)|(皮肤)|(就诊)|(血常规)|(尿常规)|(MRI)|(MRCP)|(CT)|(成像)|(检测)|(测定)|([三四五六七八九]项)', t):
            continue

        if re.search('(痛)|(腹)|(疼痛)|(进食)|(活动)|(体位)|(大便)|(排气)|(加重)|(减轻)|(坐)|(站)|(抽搐)', t):
            results.append(t)
            continue

        for feature in features:
            if re.search(feature['regex'], t, re.I):
                results.append(t)
                added = True
                break
        if added:
            continue

        ### 时间
        for regex in [time_re, re2, re3, re4, re5]:
            if re.search(regex, t, re.I):
                results.append(t)
                added = True
                break
        if added:
            continue

        for match in re.finditer(re1, t):
            if match is not None:
                match_str = match.group(0)
                if not re.match(re1_n + '{%d}' % len(match_str), match_str):
                    results.append(t)
                    added = True
                    break
        if added:
            continue

    return '，'.join(results)


def filt_data(text):
    """
    过滤非腹痛数据，非腹痛数据，诊断类型为？
    """
    if re.search('(胃|腹|腰|肋|肝|脾|肾|(剑突下)|(附件区)|(脐周)|(麦氏点))' + '[^，；。棘突脊椎]*?' + '((不适)|痛|(坠?胀))', text) or \
        re.search('肛门坠胀', text):
        return True
    else:
        return False

def read_input(input_path):
    input_data, item = [], {}
    with open(input_path, "r", encoding="utf-8") as f:
        line_ct = 0
        for line in f.readlines():
            arr = line.strip().split('\t')
            # # txt = arr[0]
            # # if len(txt) > 0 and txt[-1] not in ['。', '，']:
            #     # txt = txt + '，'
            # txt = ''
            # # txt = txt + preprocess_txt1(arr[1])
            # txt = txt + arr[1]
            # if len(txt) > 0 and txt[-1] not in ['。', '，']:
            #     txt = txt + '，'
            # # txt = txt + preprocess_txt2(arr[2])
            # txt = txt + arr[2]
            # txt = txt.strip().replace(',', '，')
            if not filt_data(arr[0] + arr[1] + arr[2]):
                continue

            txt = process_text_regex(arr[1] + '。' + arr[2])

            match_item = {'match_str': arr[3].strip().replace(',', '，'), 'label': arr[4]}

            if 'txt' in item:
                if item['txt'] == txt:
                    item['match_list'].append(match_item)
                else:
                    input_data.append(item)
                    item = {
                        'txt': txt,
                        'match_list': [match_item]
                    }
            else:
                item = {
                    'txt': txt,
                    'match_list': [match_item]
                }
            line_ct = line_ct + 1

        input_data.append(item)
        print('data length: %d->%d' % (line_ct, len(input_data)))
        logging.debug('%s' % input_data)

        with open('合并后数据3.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(input_data, indent=1, separators=(',', ':'), ensure_ascii=False))

    return input_data


def match_sub_str(text, sub_str):
    for k in range(2, len(sub_str) - 1):
        re_sub_str = sub_str[:k].replace('+', '\+').replace('-', '\-').replace('(', '\(').replace(')', '\)') + '(.*)' + sub_str[k:].replace('+', '\+').replace('-', '\-').replace('(', '\(').replace(')', '\)')
        match = re.search(re_sub_str, text)
        if match is not None:
            return match.span()
        else:
            # print(re_sub_str)
            pass

    return None, None


def get_match_str_span(text, match_str):
    match_arr = re.split('[，。；]', match_str)
    pos_arr = []
    for ms in match_arr:
        if ms == '':
            continue

        start = text.find(ms)
        if start == -1:
            start, end = match_sub_str(text, ms)
            # if start is None:
            #     raise TypeError('bad data line, not matched sub str: [%s ----- %s]' % (text, ms))
        else:
            end = start + len(ms)
        pos_arr.append((start, end))


    starts = [s for s,e in pos_arr if s is not None]
    ends = [e for s,e in pos_arr if e is not None]

    if len(starts) == 0 or len(ends) == 0:
        raise TypeError('bad data line, not matched sub str: [%s ----- %s]' % (text, match_str))

    start, end = min(starts), max(ends)

    # print(match_arr)
    # print(pos_arr)
    # print(start, end)

    return start, end


def label_data(input_data):
    # 训练数据生成
    train_data = []
    for item in input_data:
        for match_item in item["match_list"]:
            match_str, label = match_item["match_str"], match_item["label"]
            start, end = get_match_str_span(item["txt"], match_item["match_str"])
            result = {
                "content": item["txt"],
                "result_list": [{"text": item["txt"][start:end], "start": start, "end": end}],
                "prompt": label
            }
            train_data.append(result)

    return train_data


def write_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False)  + '\n')


def process(input_data, train_dev_split=True):
    # 对原始病历文本随机拆分训练、测试集，保证原始文本不会同时出现在训练和测试集中。
    # random.shuffle(input_data)
    train_input, dev_input = [], []
    print(train_dev_split)
    if train_dev_split:
        data_len = len(input_data)
        train_len = int(data_len * 0.8)
        train_input, dev_input = input_data[:train_len], input_data[train_len:]
    else:
        train_input, dev_input = input_data, []

    train_data = label_data(train_input)
    dev_data = label_data(dev_input)

    logging.debug('%s' % train_data)

    write_data(train_data, 'train.txt')
    write_data(dev_data, 'dev.txt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        help=
        '输入文件路径')
    parser.add_argument('--train_dev_split',
                        type=bool,
                        default=False,
                        help='Whether Split data for training development')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Whether Split data for training development')


    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=1)


    input_data = read_input(args.input)
    # process(input_data, args.train_dev_split)


#

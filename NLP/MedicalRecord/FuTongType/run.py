import re
import json

time_re = '(([0-9]{2,4}[-/年.])?[0-9]{1,2}[-/月.][0-9]{1,2})|([0-9]{2,4}[-/年.][0-9]{1,2}[-/月.]([0-9]{1,2})?)|([0-9]{1,2}[:：]([0-9]{2}))'
re1 = '[今凌前当昨半上中下年月日周时分旬早晚晨午夜天点时间数余近期小两号个岁一二三四五六七八九十0-9+]{2,}'
re1_n = '[一二三四五六七八九十0-9+]'
re2 = '(近来)|(傍晚)|(小时)|(去年)|([^出]现)|今|(翌日)|(长期)|(次日)|(晨起)|(既往)|((([早午晚][饭餐])|(进食.*))[前])'
re3 = '[0-9:-]((AM)|(am)|h|(PM)|(pm))'
re4 = '于.*?左右'
re5 = '(后)|(很快)|(再次)|(再发)'

def check_time_str(text):
    spans1 = []
    for regex in [time_re, re2, re3, re4]:
        for match in re.finditer(regex, text):
            if match is not None:
                spans1.append(match.span())

    for match in re.finditer(re1, text):
        if match is not None:
            match_str = match.group(0)
            if not re.match(re1_n + '{%d}' % len(match_str), match_str):
                spans1.append(match.span())

    spans2 = []
    for match in re.finditer(re5, text):
        if match is not None:
            spans2.append(match.span())

    # spans1 = merge_spans(sorted(spans1, key=lambda x: x[0]))
    # spans2 = merge_spans(sorted(spans2, key=lambda x: x[0]))
    #
    # return spans1, spans2
    # spans = merge_spans(text, sorted(spans1 + spans2, key=lambda x: x[0]))
    spans = spans1 + spans2
    return spans

def merge_spans(text, spans):
    if len(spans) == 0:
        return []
    else:
        result = []
        for idx in range(0, len(spans)):
            start, end = spans[idx][0], spans[idx+1][0] if idx < len(spans) - 1 else len(text)
            if '痛' in text[start:end]:
                result.append((start, end))

        return result


if __name__ == "__main__":
    with open('合并后数据3.json', encoding="utf-8") as f:
        json_data = json.load(f, strict=False)

    not_same_ct = 0
    for item in json_data:
        text = item['txt']
        # if len(item['match_list']) >= 2 and len(list(set([e['label'] for e in item['match_list']]))) >= 2:
        ct1 = len(item['match_list'])

        spans = check_time_str(text)
        print(text)
        print(item['match_list'])
        print('spans........')
        for start, end in spans:
            print(text[start:end])
        print('')
        ct2 = len(spans)
        not_same_ct = not_same_ct + 1 if ct1 != ct2 else not_same_ct + 0

    print(not_same_ct, len(json_data))

#

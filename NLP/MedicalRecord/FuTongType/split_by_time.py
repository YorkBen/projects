import re


# 时于点后余小左右下间中食许.约进个起两院十再

# 前昨今凌上
#
#

if __name__ == '__main__':
    time_re = '(([0-9]{2,4}[-/年.])?[0-9]{1,2}[-/月.][0-9]{1,2})|([0-9]{2,4}[-/年.][0-9]{1,2}[-/月.]([0-9]{1,2})?)|([0-9]{1,2}[:：]([0-9]{2}))'
    re1 = '[今凌前当昨半上中下年月日周时分旬早晚晨午夜天点时间数余近期小两号个岁一二三四五六七八九十0-9+]{2,}'
    re1_n = '[一二三四五六七八九十0-9+]'
    re2 = '(近来)|(再次)|(傍晚)|后|(再发)|(小时)|(去年)|现|今|(翌日)|(起初)|(逐渐)|(长期)|(随后)|(次日)|(晨起)|(最开始)|(既往)|(很快)|([早午晚][饭餐]前)'
    re3 = '[0-9:-]((AM)|(am)|h|(PM)|(pm))'
    re4 = '于.*?左右'

    ct = 0
    with open('时间词.txt', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if re.search(time_re, line):
                # print('matched!', re.search(time_re, line).group(0), line)
                pass
            elif re.search(re1, line) and not re.match(re1_n + '{%d}' % len(re.search(re1, line).group(0)), re.search(re1, line).group(0)):
                # print('matched!', re.search(re1, line).group(0), line)
                pass
            elif re.search(re2, line):
                # print('matched!', re.search(re2, line).group(0), line)
                pass
            elif re.search(re3, line):
                # print('matched!', re.search(re3, line).group(0), line)
                pass
            elif re.search(re4, line):
                # print('matched!', re.search(re3, line).group(0), line)
                pass
            else:
                ct = ct + 1
                print('not matched!', line)
                # print(line.strip())

    print('not matched: %d' % ct)

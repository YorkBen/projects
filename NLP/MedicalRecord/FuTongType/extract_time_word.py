
char_ct = {}
with open('时间词.txt', encoding="utf-8") as f:
    for line in f.readlines():
        for s in line.strip():
            if s not in char_ct:
                char_ct[s] = 0
            char_ct[s] = char_ct[s] + 1

char_arr = [(c, ct) for c, ct in char_ct.items()]
char_arr = sorted(char_arr, key=lambda x: x[1], reverse=True)
ss = ''
for c, ct in char_arr:
    if ct > 10 and c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':', '/', '一', '二', '三', '四']:
        ss = ss + c
print(ss)

前日天时月于昨今午点晨年后余小晚左右下近来上凌半周间夜中食许.约进个起两院早十再

import json
import re

with open('合并后数据32.json', encoding="utf-8") as f:
    json_data = json.load(f, strict=False)

print(len(json_data))

txt_data1, txt_data = [], []
with open('训练数据.txt', encoding="utf-8") as f:
    for line in f.readlines():
        arr = line.strip().split('\t')
        if len(txt_data1) == 0 or txt_data1[-1] != arr[0]:
            txt_data1.append(arr[0])
            txt_data.append(arr[1])

print(len(txt_data))

for j, t in zip(json_data, txt_data):
    if t.strip() == '':
        continue

    txts = re.split('，', j['txt'])
    for idx in range(len(txts)-1, -1, -1):
        if t.find(txts[idx]) >= 0:
            continue
        else:
            if idx == len(txts) - 1:
                print(txts, t, idx)
            else:
                txts[idx+1] = '今日检查' + txts[idx+1]
                break

    j['txt'] = '，'.join(txts)

with open('合并后数据33.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(json_data, indent=1, separators=(',', ':'), ensure_ascii=False))

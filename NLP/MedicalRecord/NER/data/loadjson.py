import json
import random

# 加载json数据
json_data = ''
with open(r'project.json') as f:
    json_data = json.load(f, strict=False)
    # print(json.dumps(json_data[0], indent=1, separators=(',', ':'), ensure_ascii=False))


source_result_arr = []
target_result_arr = []
result_arr = []
for item in json_data:
    raw_txt = item['data']['text'].strip()
    raw_lbl = ['O' for c in raw_txt]
    # 文字
    str1 = ' '.join(raw_txt)
    # source_result_arr.append(str)
    # print(str)

    # 标记
    for ann in item['annotations'][0]['result']:
        if ann['type'] == 'labels':
            raw_lbl[ann['value']['start']] = 'B_' + ann['value']['labels'][0]
            for i in range(ann['value']['start'] + 1, ann['value']['end']):
                raw_lbl[i] = 'I_' + ann['value']['labels'][0]

    str2 = ' '.join(raw_lbl)
    # target_result_arr.append(str)
    # print(str)
    result_arr.append((str1, str2))


random.shuffle(result_arr)

train_arr = result_arr[:-10]
dev_arr = result_arr[-10:]

with open(r'ner_data\train\source.txt', 'w') as f:
    for source, _ in train_arr:
        f.write('%s\n' % source)

with open(r'ner_data\train\target.txt', 'w') as f:
    for _, target in train_arr:
        f.write('%s\n' % target)

with open(r'ner_data\dev\source.txt', 'w') as f:
    for source, _ in dev_arr:
        f.write('%s\n' % source)

with open(r'ner_data\dev\target.txt', 'w') as f:
    for _, target in dev_arr:
        f.write('%s\n' % target)

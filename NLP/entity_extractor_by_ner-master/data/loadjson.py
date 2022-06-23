import json
import random

# 加载json数据
json_data = ''
with open(r'project-2.json') as f:
    json_data = json.load(f, strict=False)
    # print(json.dumps(json_data[0], indent=1, separators=(',', ':'), ensure_ascii=False))


data_arr = []
for item in json_data:
    raw_txt = item['data']['text'].strip()
    raw_lbl = ['O' for c in raw_txt]
    # 文字
    # str = ''.join(raw_txt)
    # source_result_arr.append(str)
    # print(str)

    # 标记
    for ann in item['annotations'][0]['result']:
        if ann['type'] == 'labels':
            raw_lbl[ann['value']['start']] = 'B_' + ann['value']['labels'][0]
            for i in range(ann['value']['start'] + 1, ann['value']['end']):
                raw_lbl[i] = 'I_' + ann['value']['labels'][0]

    data_arr.append((raw_txt, raw_lbl))
    # str = ''.join(raw_lbl)
    # target_result_arr.append(str)
    # print(str)

random.shuffle(data_arr)

# train_set = data_arr[:-10]
# dev_set = data_arr[-10:]
# test_set = data_arr[-1:]

def write_data(arr, f):
    line_ct, o_ct = 0, 0
    for str, lbl in arr:
        line_ct = line_ct + 1
        for s, l in zip(str, lbl):
            f.write('%s %s\n' % (s, l))
            if s == '。':
                o_ct = o_ct + 1
                f.write('\n')
        if s != '。':
            f.write('\n')
    print(line_ct, o_ct)

with open('train.txt', 'w') as f:
    write_data(data_arr, f)

# with open('dev.txt', 'w') as f:
#     write_data(dev_set, f)

# with open('test.txt', 'w') as f:
#     write_data(test_set, f)

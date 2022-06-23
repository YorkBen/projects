import json
import random

# 加载json数据
json_data = ''
with open(r'project.json') as f:
    json_data = json.load(f, strict=False)
    print(json.dumps(json_data[0], indent=1, separators=(',', ':'), ensure_ascii=False))

exit()

data_arr = []
for item in json_data:
    text_dict = {}
    relations = []
    # 标记
    for ann in item['annotations'][0]['result']:
        if ann['type'] == 'labels':
            text_dict[ann['id']] = ann
        elif ann['type'] == 'relation':
            if ann['direction'] == 'right':
                relations.append((ann['from_id'], ann['to_id']))
            else:
                relations.append((ann['to_id'], ann['from_id']))


    raw_txt = item['data']['text'].strip()
    raw_lbl = ['O' for c in raw_txt]
    # 文字
    # str = ''.join(raw_txt)
    # source_result_arr.append(str)
    # print(str)



    data_arr.append((raw_txt, raw_lbl))
    # str = ''.join(raw_lbl)
    # target_result_arr.append(str)
    # print(str)

random.shuffle(data_arr)

train_set = data_arr[:-11]
dev_set = data_arr[-11:-1]
test_set = data_arr[-1:]

def write_data(arr, f):
    for str, lbl in arr:
        for s, l in zip(str, lbl):
            f.write('%s %s\n' % (s, l))
            if s == '。':
                f.write('\n')
        if s != '。':
            f.write('\n')


with open('input/train.txt', 'w') as f:
    write_data(train_set, f)

with open('input/dev.txt', 'w') as f:
    write_data(dev_set, f)

with open('input/test.txt', 'w') as f:
    write_data(test_set, f)

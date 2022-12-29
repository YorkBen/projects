import json

with open('合并后数据32.json', encoding="utf-8") as f:
    json_data = json.load(f, strict=False)

for idx, item in enumerate(json_data):
    if item['txt'] == '者10天前无明显诱因出现左下腹部疼痛，持续疼痛，不伴右腹部疼痛，10天前无明显诱因右侧背部开始起红疹、水疱，无右侧腹部疼痛，大便未解，体力体重较前明显减轻':
        print(idx)

exit()
with open('tmp.txt', encoding="utf-8") as f:
    for line in f.readlines():
        arr = line.strip().split('\t')
        matched = False
        for item in json_data:
            for match_item in item["match_list"]:
                if match_item["match_str"].find(arr[0]) != -1:
                    match_item["label"] = arr[1]
                    matched = True
                    break
            if matched:
                break

        if not matched:
            print('%s not matched!' % arr[0])

with open('合并后数据32.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(json_data, indent=1, separators=(',', ':'), ensure_ascii=False))

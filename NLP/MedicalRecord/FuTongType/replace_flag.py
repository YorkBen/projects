import json

with open('合并后数据3.json', encoding="utf-8") as f:
    json_data = json.load(f, strict=False)

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

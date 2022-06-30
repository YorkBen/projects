import json
import random


def load_regex():
    """
    返回文字标签字典
    """
    json_data = ''
    with open(r'project.json') as f:
        json_data = json.load(f, strict=False)
        # print(json.dumps(json_data[0], indent=1, separators=(',', ':'), ensure_ascii=False))
    result_arr = []
    for item in json_data:
        for ann in item['annotations'][0]['result']:
            if ann['type'] == 'labels' and ann['origin'] == 'manual':
                result_arr.append([ann['value']['text'], ann['value']['labels'][0], 1])

    result_arr = sorted(result_arr, key=lambda x: x[0] + x[1])
    result_dict = {}
    for item in result_arr[1:]:
        key = item[0] + '_' + item[1]
        if key not in result_dict:
            result_dict[key] = 1
        else:
            result_dict[key] = result_dict[key] + 1

    result_arr = []
    for key in result_dict.keys():
        tmp = key.split('_')
        result_arr.append((tmp[0], tmp[1], result_dict[key]))

    # result_arr -> text, label, count
    result_arr = sorted(result_arr, key=lambda x: (len(x[0]), x[2]), reverse=True)

    result_dict = {}
    for r in result_arr:
        if r[0] not in result_dict:
            result_dict[r[0]] = r[1]

    return result_dict


def process_text_backup(text, text_label_dict):
    text_label_result = []
    for key in text_label_dict.keys():
        pos, key_len = -1, len(key)
        while True:
            pos = text.find(key, pos+1)
            # print(key, pos)
            if pos == -1:
                break
            text_label_result.append({
                "start": pos,
                "end": pos + key_len,
                "text": key,
                "score": 0.50,
                "labels":[
                    text_label_dict[key]
                ]
            })
        text = text.replace(key, ' ' * key_len)
    return text_label_result


def process_text(text, text_label_dict):
    text_label_result = []
    for idx in range(len(text)):
        for key in text_label_dict.keys():
            if text[idx:].startswith(key):
                text_label_result.append({
                    "start": idx,
                    "end": idx + len(key),
                    "text": key,
                    "score": 0.50,
                    "labels":[
                        text_label_dict[key]
                    ]
                })
                text = text.replace(key, ' ' * len(key), 1)
                break

    return text_label_result


def wrap_result(project_name, text_no, text, text_label_result):
    """
    """
    result = []
    for idx, item in enumerate(text_label_result):
        result.append({
            "id": "%s_%s_%s" % (project_name, text_no, idx),
            "from_name":"label",
            "to_name":"text",
            "type":"labels",
            "value": item
        })

    return {
        "data": {
            "text": text
        },
        "predictions": [{
            "model_version": "one",
            "result": result
        }]
    }

if __name__ == "__main__":
    project_name = 'mr2000'
    text_label_dict = load_regex()

    res_arr = []
    with open(r'../FeatureExtraction/data/labeled_ind.txt') as f:
        for text_no, line in enumerate(f.readlines()):
            if text_no >= 108:
                print('processing line %s' % text_no)
                text = line.strip().split('	')[1]
                res = process_text(text, text_label_dict)
                res = wrap_result(project_name, text_no, text, res)
                res_arr.append(res)

    with open(r'result.json', "w") as f:
        f.write(json.dumps(res_arr, indent=1, separators=(',', ':'), ensure_ascii=False))


#

import json
import random
import re
from time import time

def load_nerlbl_from_manual(json_file):
    """
    从LabelStudio的导出数据中，抓取人工标注的命名实体数据，即词和命名实体类别对。
    """
    json_data = ''
    with open(json_file) as f:
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


def label_line_by_regex(text, text_label_dict, language='ZH'):
    """
    对一行文本通过正则来进行标记
    """
    text_label_result = []
    if language == 'ZH':
        ct = 0
        for idx in range(len(text)):
            for key in text_label_dict.keys():
                if text[idx:].startswith(key):
                    text_label_result.append((ct, idx, idx+len(key), key, text_label_dict[key]))
                    text = text.replace(key, ' ' * len(key), 1)
                    ct = ct + 1
                    break
    elif language == 'EN':
        ct = 0

        ## 多个regex
        for key, label in text_label_dict.items():
            if ' ' in key:
                matches = re.finditer('(^| )%s( |$|\.|,)' % key, text, re.I)
            else:
                matches = re.finditer('(^| )%s( |$|\.|,)' % key, text)
                
            for match in matches:
                start, end = match.span()
                match_str = match.group(0)
                if match_str[0] == ' ':
                    start = start + 1
                    match_str = match_str[1:]
                if match_str[-1] in ['.', ',', ' ']:
                    match_str = match_str[:-1]
                    end = end - 1

                text_label_result.append((ct, start, end, match_str, label))
                ct = ct + 1

    return text_label_result


def process_by_regex(text_label_dict, texts, language='ZH'):
    """
    通过正则表达式来进行文本自动标记
    """
    res_arr = []
    for line_no, text in enumerate(texts):
        time_start = time()
        print('processing line: %s' % (line_no+1))
        text = text.strip()
        labels = label_line_by_regex(text, text_label_dict, language)
        entities = []
        for id, start, end, word, label in labels:
            entities.append(assemble_ner_entity('%s_%s' % (line_no, id), start, end, word, label))
        res_arr.append(assemble_ner_result(text, entities))
        time_end = time()
        print('time cost: %s' % (time_end - time_start))
    return res_arr


def process_by_predict(json_file, input_file):
    """
    从机器预测的NER结果来进行文本标注，生成LabelStudio导入数据。
    输入：json数组，数组每行为每行的所有标注：[[entity_type, [start, end]], ...]
    """
    json_data = ''
    with open(json_file) as f:
        json_data = json.load(f, strict=False)

    res_arr = []
    with open(input_file) as f:
        for line_no, text in enumerate(f.readlines()):
            print('processing line: %s' % (line_no+1))
            text = text.strip()
            labels = json_data[line_no]
            entities = []
            for id, l in enumerate(labels):
                label, start, end = l[0], l[1][0], l[1][1]
                word = text[start:end]
                entities.append(assemble_ner_entity('%s_%s' % (line_no, id), start, end, word, label))
            entities.append({
                "from_id": "%s_%s" % (line_no, 0),
                "to_id": "%s_%s" % (line_no, 1),
                "type": "relation",
                "direction": "right",
                "labels":[]
            })
            res_arr.append(assemble_ner_result(text, entities))

    return res_arr


def assemble_ner_entity(id, start, end, text, label):
    """
    组装命名实体单个entity结果json
    """
    return {
        "id": id,
        "from_name":"label",
        "to_name":"text",
        "type":"labels",
        "value": {
            "start": start,
            "end": end,
            "text": text,
            "score": 0.50,
            "labels":[
                label
            ]
        }
    }

def assemble_ner_result(text, entities):
    """
    组装NER结果json
    """
    return {
        "data": {
            "text": text
        },
        "predictions": [{
            "model_version": "one",
            "result": entities
        }]
    }



if __name__ == "__main__":
    ## 单词标签数据
    # text_label_dict = load_nerlbl_from_manual(json_file)

    ## 文本数据
    # texts = []
    # with open(input_file) as f:
    #     for line in f.readlines():
    #         texts.append(line.strip())

    # res_arr = process_by_regex(text_label_dict, r'../FeatureExtraction/data/labeled_ind.txt')
    # res_arr = process_by_predict(r'../entity_extractor_by_ner-master/data/predict_result.json', r'../entity_extractor_by_ner-master/data/test.txt')

    ## 写结果
    # with open(r'result.json', "w") as f:
    #     f.write(json.dumps(res_arr, indent=1, separators=(',', ':'), ensure_ascii=False))
    pass

#

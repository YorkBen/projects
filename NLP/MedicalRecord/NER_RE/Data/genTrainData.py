import json
import random
import argparse
import re
import math
import copy

"""
从LabelStudio导出的json数据中生成NER训练数据或RE训练数据
运行参数：
    -i 输入json文件
    -o 输出文件名
    -t 生成数据类型：NER为命名实体识别，RE为关系抽取
"""

def write_ner_data(arr, file_path):
    """
    写NER数据，格式为字符空格字符BIO标记
    输入格式：每行语句的文本和标记对数组
    """
    with open(file_path, 'w') as f:
        for str, lbl in arr:
            for s, l in zip(str, lbl):
                f.write('%s %s\n' % (s, l))
                if s == '。':
                    f.write('\n')
            if s != '。':
                f.write('\n')


def write_lines(arr, file_path):
    """
    将数据行写入文件
    """
    with open(file_path, 'w') as f:
        for l in arr:
            f.write('%s\n' % l)


def sample_negtive_ids(id_list, key_id, pos_ids, ratio=1):
    """
    关系标记只有正例标记，没有负例标记，从所有的id列表中按照采样比例选取不在正例id列表里面的
    id_list: [(id, start)]按照start顺序排列的数组
    pos_ids: [id]
    ratio: 负例相对正例采样比例
    返回：负例id列表
    """
    len_pos = len(pos_ids)
    sample_num = round(ratio * len_pos)
    ids = [e[0] for e in id_list]
    ids.remove(key_id)
    candidate_ids = list(set(ids).difference(set(pos_ids)))
    sample_num = len(candidate_ids) if sample_num > len(candidate_ids) else sample_num

    return random.sample(candidate_ids, sample_num)


def get_entities(item):
    """
    从LabelStudio中导出的数据中抓取实体数据
    """
    # id -> NER标记
    ann_dict, id_list = {}, []
    for ann in item['annotations'][0]['result']:
        if ann['type'] == 'labels':
            ann_dict[ann['id']] = ann
            id_list.append((ann['id'], ann['value']['start']))

    id_list = sorted(id_list, key=lambda x: x[1])

    return ann_dict, id_list


def get_relations(item):
    """
    从LabelStudio中导出的数据中抓取关系数据
    """
    # 合并整理关系标记，加入正例和反例
    relations = []
    for ann in item['annotations'][0]['result']:
        if ann['type'] == 'relation':
            label = '1' if len(ann["labels"]) == 0 else ann["labels"][0]
            if ann['direction'] == 'right':
                from_id, to_id = ann['from_id'], ann['to_id']
            else:
                from_id, to_id = ann['to_id'], ann['from_id']

            relations.append((from_id, to_id, label))

    return relations


def expand_neg_samples(relations, id_list, ratio):
    """
    关系列表扩充负例数据：
    1. 从标记的from_id -> to_ids中，增加from_id -> 非标记to_id对
    2. 从标记的to_id -> from_ids中，增加非标记from_id -> to_id对
    3. 合并数据&去重
    """
    rel_from, rel_to = {}, {}
    for from_id, to_id, _ in relations:
        # from作为key
        if from_id not in rel_from:
            rel_from[from_id] = []
        rel_from[from_id].append(to_id)

        # to作为key
        if to_id not in rel_to:
            rel_to[to_id] = []
        rel_to[to_id].append(from_id)

    # 构造反例数据集
    # positive from_id negtive to_id
    rel_neg_to = {}
    for from_id in rel_from.keys():
        rel_neg_to[from_id] = sample_negtive_ids(id_list, from_id, rel_from[from_id], ratio)

    # negtive from_id positive to_id
    rel_neg_from = {}
    for to_id in rel_to.keys():
        rel_neg_from[to_id] = sample_negtive_ids(id_list, to_id, rel_to[to_id], ratio)

    # 合并数据集
    for from_id in rel_neg_to.keys():
        for to_id in rel_neg_to[from_id]:
            relations.append((from_id, to_id, 0))

    for to_id in rel_neg_from.keys():
        for from_id in rel_neg_from[to_id]:
            relations.append((from_id, to_id, 0))

    relations = list(set(relations))
    return relations


def assembleREData(text, ann_dict, relations):
    """
    根据关系对数据生成训练数据：
    在from_id前后加'#'，在to_id前后加'$'，并加上from_id、to_id开始结束位置以及分类标签
    """
    results = []
    for from_id, to_id, label in relations:
        # 标记开始结束位置
        from_start_pos = ann_dict[from_id]['value']['start']
        from_end_pos = ann_dict[from_id]['value']['end']
        to_start_pos = ann_dict[to_id]['value']['start']
        to_end_pos = ann_dict[to_id]['value']['end']

        if from_start_pos < to_start_pos:
            item_text = text[:from_start_pos] + '#' + text[from_start_pos:from_end_pos] + '#' + text[from_end_pos:to_start_pos] + '$' + text[to_start_pos:to_end_pos] + '$' + text[to_end_pos:]
            results.append('%s\t%s\t%s\t%s\t%s\t%s' % (item_text, from_start_pos, from_end_pos+1, to_start_pos+2, to_end_pos+3, label))
        else:
            item_text = text[:to_start_pos] + '$' + text[to_start_pos:to_end_pos] + '$' + text[to_end_pos:from_start_pos] + '#' + text[from_start_pos:from_end_pos] + '#' + text[from_end_pos:]
            results.append('%s\t%s\t%s\t%s\t%s\t%s' % (item_text, from_start_pos+2, from_end_pos+3, to_start_pos, to_end_pos+1, label))

    return results


def assembleNERData(text, ann_dict):
    """
    生成命名实体识别标记数据
    """
    raw_lbl = ['O' for c in text]

    # 标记
    for ann in ann_dict.values():
        label_id = ann['value']['labels'][0]
        raw_lbl[ann['value']['start']] = 'B_' + label_id
        for i in range(ann['value']['start'] + 1, ann['value']['end']):
            raw_lbl[i] = 'I_' + label_id

    return (text, raw_lbl)


def merge_text_length(text_lens, n):
    """
    将文本长度数组合并至长度为n
    """
    while len(text_lens) > n:
        minpos = text_lens.index(min(text_lens))
        text_lens_merge = []
        if minpos == 0:
            text_lens_merge.append(text_lens[0] + text_lens[1])
            text_lens_merge.extend(text_lens[2:])
        elif minpos == len(text_lens) - 1:
            text_lens_merge.extend(text_lens[:-2])
            text_lens_merge.append(text_lens[-2] + text_lens[-1])
        elif text_lens[minpos-1] <= text_lens[minpos+1]:
            text_lens_merge.extend(text_lens[:minpos-1])
            text_lens_merge.append(text_lens[minpos-1] + text_lens[minpos])
            text_lens_merge.extend(text_lens[minpos+1:])
        else:
            text_lens_merge.extend(text_lens[:minpos])
            text_lens_merge.append(text_lens[minpos] + text_lens[minpos+1])
            text_lens_merge.extend(text_lens[minpos+2:])

        text_lens = text_lens_merge

    return text_lens


def split_longtext_item(item):
    """
    如果标注数据文本长度过长，则将标注条目按照'。'拆分成多个。
    """
    text = item['data']['text'].strip()
    text_length, delta = len(text), 490
    if text_length < delta:
        return [item]
    else:
        print(item)
        n = int(math.ceil(text_length / delta))
        print('text length, split num：', text_length, n)
        texts = text.split('。')
        text_lens = [len(t) + 1 for t in texts[:-1]]
        print('sub text lengths：', text_lens)
        text_lens = merge_text_length(text_lens, n)
        print('merged text lengths：', text_lens)

        # 所有实体和关系
        ann_dict, id_list = get_entities(item)
        relations = get_relations(item)

        # 根据数量合并文本段
        items = []
        start, relations_num = 0, 0
        for ind, tlen in enumerate(text_lens):
            end = start + tlen
            item_ = {
                "data": {
                    "text": text[start:end]
                },
                "annotations":[{
                    "result": []
                }]
            }

            # 开始筛选实体和关系
            entities = []
            for ann in ann_dict.values():
                if ann["value"]["start"] >= start and  ann["value"]["start"] < end \
                    and ann["value"]["end"] >= start and  ann["value"]["end"] < end:
                    entities.append(ann["id"])
                    item_["annotations"][0]["result"].append(copy.deepcopy(ann))
                    item_["annotations"][0]["result"][-1]["value"]["start"] = item_["annotations"][0]["result"][-1]["value"]["start"] - start
                    item_["annotations"][0]["result"][-1]["value"]["end"] = item_["annotations"][0]["result"][-1]["value"]["end"] - start

                    # print(item_["annotations"][0]["result"][-1]["value"]["text"], \
                    #     text[start:end][item_["annotations"][0]["result"][-1]["value"]["start"]:item_["annotations"][0]["result"][-1]["value"]["end"]])

            # 关系
            for from_id, to_id, _ in relations:
                if from_id in entities and to_id in entities:
                    item_["annotations"][0]["result"].append({
                        "from_id":from_id,
                         "to_id":to_id,
                         "type":"relation",
                         "direction":"right",
                         "labels":[]
                    })
                    relations_num = relations_num + 1

            # 其它
            start = end
            items.append(item_)
        # print(items)
        print('relations num: origional:%s, after split:%s' % (len(relations), relations_num))
        return items


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
    parser.add_argument('-i', type=str, default='project.json', help='input file')
    parser.add_argument('-o', type=str, default='train_data.txt', help='output file')
    parser.add_argument('-t', type=str, default='NER', help='data type')
    parser.add_argument('-r', type=int, default=1, help='negtive sample ratio')
    args = parser.parse_args()

    input = args.i
    output = args.o
    type = args.t
    ratio = args.r

    print("input: %s, output: %s, runtype: %s, ratio: %s" % (input, output, type, ratio))
    if type not in ['NER', 'RE']:
        print('Error: parameter type must be one of [NER, RE]')
        exit()

    # 加载json数据
    json_data = ''
    with open(input) as f:
        json_data = json.load(f, strict=False)
        # print(json.dumps(json_data[0], indent=1, separators=(',', ':'), ensure_ascii=False))
        # exit()

    # 生成数据
    results = []
    for item_r in json_data:
        for item in split_longtext_item(item_r):
            text = item['data']['text'].strip()
            ann_dict, id_list = get_entities(item)
            if type == 'NER':
                results.append(assembleNERData(text, ann_dict))
            elif type == 'RE':
                relations = get_relations(item)
                relations = expand_neg_samples(relations, id_list, ratio)
                results.extend(assembleREData(text, ann_dict, relations))

    random.shuffle(results)

    # write results
    if type == 'NER':
        write_ner_data(results, output)
    elif type == 'RE':
        write_lines(results, output)

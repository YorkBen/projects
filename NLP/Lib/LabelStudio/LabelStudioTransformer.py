import json
import random

class Transformer:
    def __init__(self):
        pass

    def load_json_file(self, file_path):
        """
        加载json文件
        """
        json_data = ''
        with open(file_path, encoding='utf-8') as f:
            json_data = json.load(f, strict=False)

        return json_data

    def write_json_file(self, json_data, file_path):
        """
        写json文件
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(json_data, indent=1, separators=(',', ':'), ensure_ascii=False))


    def get_ner_label_data(self, file_path):
        """
        解析导出数据，生成NER标记数据
        """
        json_data = self.load_json_file(file_path)

        # ner
        ner_dict = {}
        for item in json_data:
            entities = self.get_entities(item)
            ner_dict |= self.get_entity_text_label(entities)

        return ner_dict


    def get_entities(self, item):
        """
        从LabelStudio中导出的数据中抓取实体数据
        """
        entities = []
        for r in item['annotations'][0]['result']:
            if r["type"] == "labels" and "text" in r['value'] and 'labels' in r['value']:
                # 有些标记前后有空格。标点符号，代码处理掉。
                text = r['value']['text']
                start = r['value']['start']
                end = r['value']['end']
                text_ = text.strip()
                if text_[-1] in ['.', ',', '，', '。', ';', '；']:
                    text_ = text_[:-1]

                start = start + text.index(text_)
                end = start + len(text_)
                text = text_

                label = r['value']['labels'][0]
                entities.append((r['id'], start, end, text, label))

        entities = sorted(entities, key=lambda x: (x[1], x[2]))

        return entities

    def merge_entities(self, entities):
        if len(entities) == 0:
            return entities

        result = []
        for k in range(len(entities) - 1):
            e1, e2 = entities[k], entities[k+1]
            if e1[1] <= e2[1] and e1[2] >= e2[2]:
                continue
            else:
                result.append(e1)

        result.append(entities[-1])
        return result


    def search_entities_in_list(self, entities, ids):
        """
        在列表中检索实体
        """
        e_dict = {}
        for e in entities:
            e_dict[e[0]] = e

        results = []
        for id in ids:
            if id in e_dict:
                results.append(e_dict[id])

        return results


    def get_entity_label_texts(self, entities):
        """
        获取实体类型->实体文字字典
        """
        lt_dict = {}
        for e in entities:
            if e[-1] not in lt_dict:
                lt_dict[e[-1]] = []
            lt_dict[e[-1]].append(e[-2])

        for l, texts in lt_dict.items():
            lt_dict[l] = list(set(texts))

        return lt_dict

    def get_entity_text_label(self, entities):
        # ner
        ner_dict = {}
        for e in entities:
            ner_dict[e[-2]] = e[-1]

        return ner_dict


    # def replace_entity_by_label(self, text, entities):
    #     """
    #     替换文本中的实体字符串为实体标签。
    #     entities是排过序的
    #     """
    #     entities = sorted(entities, key=lambda x: x[1])
    #     r_text = ''
    #     a_start = 0
    #     for _, start, end, _, label in entities:
    #         r_text = r_text + text[a_start:start]
    #         r_text = r_text + '#%s#' % label
    #         a_start = end
    #     r_text = r_text + text[a_start:]
    #
    #     # start, end = 0, len(r_text) - 1
    #     # for i in range(0, len(r_text)):
    #     #     if r_text[i] == ',':
    #     #         start = i
    #     #     if r_text[i] == '#':
    #     #         break
    #     #
    #     # for i in range(len(r_text)-1, -1, -1):
    #     #     if r_text[i] == ',':
    #     #         end = i
    #     #     if r_text[i] == '#':
    #     #         break
    #     #
    #     # return r_text[start:end+1]
    #     return r_text
    #
    #
    # def replace_entity_by_other_entities(self, text, entities, lt_dict):
    #     """
    #     替换文本中的实体字符串为实体标签。
    #     entities是排过序的
    #     lt_dict是标签到文字数组的字典
    #     """
    #     r_text = ''
    #     a_start = 0
    #     for _, start, end, _, label in entities:
    #         r_text = r_text + text[a_start:start]
    #         r_text = r_text + '#%s#' % random.choice(lt_dict[label])
    #         a_start = end
    #     r_text = r_text + text[a_start:]
    #     return r_text


    def get_relations(self, item):
        """
        从LabelStudio中导出的数据中抓取关系数据
        """
        # 合并整理关系标记，加入正例和反例
        relations = []
        for ann in item['annotations'][0]['result']:
            if ann['type'] == 'relation':
                label = 'null' if "labels" not in ann or len(ann["labels"]) == 0 else ann["labels"][0]
                if ann['direction'] == 'right':
                    from_id, to_id = ann['from_id'], ann['to_id']
                else:
                    from_id, to_id = ann['to_id'], ann['from_id']

                relations.append((from_id, to_id, label))

        return relations

    def get_entity_relations(self, relations, entities):
        """
        获取实体关系对
        """
        entity_dict = {}
        for e in entities:
            entity_dict[e[0]] = e

        results = []
        for r in relations:
            results.append((entity_dict[r[0]], entity_dict[r[1]], r[2]))

        return results


    def filter_project_json(self, json_data):
        """
        解析导出数据，去掉不需要信息
        """
        results = []
        for item in json_data:
            results.append({
                "annotations": item["annotations"],
                "data": item["data"]
            })

        return results

    def merge_project_and_manual_data(self, file_path1, file_path2):
        """
        合并导出文件file_path1和人工生成文件file_path2
        """
        project_json = self.filter_project_json(self.load_json_file(file_path1))
        manual_json = self.load_json_file(file_path2)

        for item in project_json:
            text_head = item["data"]["text"][:100]
            for idx, item2 in enumerate(manual_json):
                if item2["data"]["text"][:100] == text_head:
                    manual_json[idx] = item
                    break

        return manual_json


    def assemble_ner_entity_ann(self, id, start, end, text, label, score=0.5):
        """
        组装单个命名实体标记
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
                "score": score,
                "labels":[
                    label
                ]
            }
        }

    def assembel_relation_entity_ann(self, from_id, to_id, label):
        """
        组装单个关系标记
        """
        return {
            "from_id":from_id,
            "to_id":to_id,
            "type":"relation",
            "direction":"right",
            "labels":[label]
        }

    def assemble_anns(self, text, entities):
        """
        组装NER结果json
        """
        return {
            "data": {
                "text": text
            },
            "annotations": [{
                # "model_version": "one",
                "result": entities
            }]
        }


#

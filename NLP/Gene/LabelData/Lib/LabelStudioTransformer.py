import json


class Transformer:
    def __init__(self):
        pass

    def load_json_file(self, file_path):
        """
        加载json文件
        """
        json_data = ''
        with open(file_path) as f:
            json_data = json.load(f, strict=False)

        return json_data

    def write_json_file(self, json_data, file_path):
        """
        写json文件
        """
        with open(file_path, "w") as f:
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
            ner_dict |= self.get_entity_labeled_data(entities)

        return ner_dict


    def get_entities(self, item):
        """
        从LabelStudio中导出的数据中抓取实体数据
        """
        entities = []
        for r in item['annotations'][0]['result']:
            if r["type"] == "labels":
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

        return entities

    def get_entity_labeled_data(self, entities):
        # ner
        ner_dict = {}
        for e in entities:
            ner_dict[e[-2]] = e[-1]

        return ner_dict

    def get_relations(self, item):
        """
        从LabelStudio中导出的数据中抓取关系数据
        """
        # 合并整理关系标记，加入正例和反例
        relations = []
        for ann in item['annotations'][0]['result']:
            if ann['type'] == 'relation':
                label = '1' if "labels" not in ann or len(ann["labels"]) == 0 else ann["labels"][0]
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





#

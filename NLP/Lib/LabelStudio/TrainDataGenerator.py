import json
import random
import argparse
import re
import math
import copy

from LabelStudioTransformer import Transformer
from TrainDataBase import TrainDataBase

class TrainDataGenerator(TrainDataBase):
    def __init__(self):
        self.ls_transfomer = Transformer()

    def split_item_by_sep(self, item, separator='\n', append_sep=False):
        """
        根据指定符号来拆分标记数据。
        """
        text = item['data']['text']
        texts = text.split(separator)
        if len(texts) == 1:
            return [item]
        else:
            # print('split_item_by_sep: %s' % separator)
            text_lens = [len(t) for t in texts]
            # print('split num: %s' % len(texts))

            # 所有实体和关系
            entities = self.ls_transfomer.get_entities(item)
            relations = self.ls_transfomer.get_relations(item)

            # 根据数量合并文本段
            items = []
            start, sep_append_len = 0, len(separator.strip())
            for tlen in text_lens:
                end = start + tlen + sep_append_len

                # 过滤空行
                if len(text[start:end]) > 2:
                    item_ = self.gen_splited_item(text, start, end, entities, relations)
                    items.append(item_)

                start = end + (len(separator) - sep_append_len) # 加上\n偏置
            # print('relations num: origional:%s, after split:%s' % (len(relations), relations_num))
            return items


    def split_item_banlance_txtlength(self, item, delta=490, language='CH'):
        """
        根据文本长度参数拆分标注数据，均衡文本长度。
        """
        text = item['data']['text'].strip()
        text_length = len(text)
        if text_length < delta:
            return [item]
        else:
            sent_sep = '。' if language == 'CH' else '. '
            texts = text.split(sent_sep)
            text_lens = [len(t) + len(sent_sep) for t in texts[:-1]]
            if len(texts[-1].strip()) > 2:
                 text_lens.append(len(texts[-1]))
            # print('sub text lengths：', text_lens)
            text_lens = self.merge_text_length(text_lens, delta)
            # print('merged text lengths：', text_lens)

            # 所有实体和关系
            entities = self.ls_transfomer.get_entities(item)
            relations = self.ls_transfomer.get_relations(item)

            # 根据数量合并文本段
            items, start = [], 0
            for ind, tlen in enumerate(text_lens):
                end = start + tlen
                item_ = self.gen_splited_item(text, start, end, entities, relations)
                # print(item_)

                # 其它
                start = end
                items.append(item_)
            # print(items)
            # print('relations num: origional:%s, after split:%s' % (len(relations), relations_num))
            return items


    def merge_text_length(self, text_lens, delta):
        """
        按长度参数合并文本，两个长度的和小于参数则合并
        """
        result_lens, idx = [text_lens[0]], 1
        while idx < len(text_lens):
            if result_lens[-1] + text_lens[idx] <= delta:
                result_lens[-1] = result_lens[-1] + text_lens[idx]
            else:
                result_lens.append(text_lens[idx])
            idx = idx + 1

        return result_lens


    def gen_splited_item(self, text, start, end, entities, relations):
        """
        根据指定的开始和结束位置拆分一条标记数据。
        entities: [(id, start, end, text, label)]
        relations: [(from_id, to_id, label)]
        """
        # 开始筛选实体和关系
        anns, entity_ids = [], []
        for e in entities:
            if e[1] >= start and e[1] < end \
                and e[2] > start and e[2] <= end:
                entity_ids.append(e[0])

                entity_ann = self.ls_transfomer.assemble_ner_entity_ann(*e)
                entity_ann["value"]["start"] = entity_ann["value"]["start"] - start
                entity_ann["value"]["end"] = entity_ann["value"]["end"] - start
                anns.append(entity_ann)

        # 关系
        for from_id, to_id, relation in relations:
            if from_id in entity_ids and to_id in entity_ids:
                anns.append(self.ls_transfomer.assembel_relation_entity_ann(from_id, to_id, relation))


        return self.ls_transfomer.assemble_anns(text[start:end], anns)


    def gen_ner_data(self, text, entities, key_labels=None, include_labels=None, language='CH'):
        """
        生成命名实体识别标记数据
        entities: [(id, start, end, text, label)]
        key_labels: ['a', 'b', 'c']
        include_labels: ['a', 'b', 'c']
        """
        if len(entities) == 0:
            return None

        raw_lbl = ['O' for c in text]

        # 判断是否包含了key_label，或者无label，如果没有则不输出该句。
        all_labels = [e[4] for e in entities]
        if key_labels is not None and \
                len(set(key_labels).intersection(set(all_labels))) == 0:
            return None

        # 标记
        proc_labels = ([] if key_labels is None else key_labels) + ([] if include_labels is None else include_labels)
        for id, start, end, e_text, label in entities:
            if len(proc_labels) == 0 or label in proc_labels:
                raw_lbl[start] = 'B_' + label.replace(' ', '-')
                for i in range(start+1, end):
                    raw_lbl[i] = 'I_' + label.replace(' ', '-')

        # # 用于记录有多少个不同标记
        # label_list = all_labels if len(proc_labels) == 0 else list(set(proc_labels).intersection(set(all_labels)))

        # 英文标注按照空格合并替换
        if language == 'EN':
            words, labels = [], []
            start, ind = 0, 0
            while ind < len(text):
                c, l = text[ind], raw_lbl[ind]
                if c in [',', '.', ' ', '!', '?', ':', '\'', '\"']:
                    if c == '.' and ind != len(text) - 1 and text[ind+1] != ' ':
                        ind = ind + 1
                        continue
                    if text[start:ind] != ' ':
                        words.append(text[start:ind])
                        labels.append(raw_lbl[start])
                    if c != ' ':
                        start = ind
                    else:
                        start = ind + 1
                ind = ind + 1
            return words, labels

        return (text, raw_lbl)


    def balance_ner_data(self, data, strategy):
        """
        均衡NER数据。
        输入格式为gen_ner_data函数输出的数组：[(text, label_list)]
        """
        label_data_dict = {}
        for text, label_list in data:
            label_set = set(label_list)
            label_set.remove('O')
            label_list_ = list(label_set)
            labels = list(set([l.replace('B_', '').replace('I_', '') for l in label_list_]))
            for label in labels:
                if label not in label_data_dict:
                    label_data_dict[label] = []
                label_data_dict[label].append((text, label_list))

        return self.balance_data(label_data_dict, strategy=strategy)

    def balance_ner_data_forspacy(self, data, strategy):
        """
        均衡NER数据。
        输入格式为gen_ner_data函数输出的数组：[(text, label_list)]
        """
        label_data_dict = {}
        for text, label_list in data:
            label_set = set([e[2] for e in label_list])
            labels = list(label_set)
            for label in labels:
                if label not in label_data_dict:
                    label_data_dict[label] = []
                label_data_dict[label].append((text, label_list))

        return self.balance_data(label_data_dict, strategy=strategy)


    def write_ner_data(self, data, file_path):
        """
        写NER数据，格式为字符空格字符BIO标记
        输入格式：每行语句的文本和标记对数组
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for str, lbl in data:
                for s, l in zip(str, lbl):
                    # 写空格出错
                    if s.strip() == '':
                        continue
                    f.write('%s %s\n' % (s, l))
                f.write('\n')

    def expand_neg_relations(self, entities, relations, ratio=0.2):
        """
        关系列表扩充负例数据：
        1. 从标记的from_id -> to_ids中，增加from_id -> 非标记to_id对
        2. 从标记的to_id -> from_ids中，增加非标记from_id -> to_id对
        3. 合并数据&去重
        """
        def sample_negtive_ids(ids, key_id, val_ids, ratio=1):
            """
            关系标记只有正例标记，没有负例标记，从所有的id列表中按照采样比例选取不在正例id列表里面的
            id_list: [(id, start)]按照start顺序排列的数组
            pos_ids: [id]
            ratio: 负例相对正例采样比例
            返回：负例id列表
            """
            ids.remove(key_id)
            candidate_ids = list(set(ids).difference(set(val_ids)))

            sample_num = round(ratio * len(val_ids))
            sample_num = len(candidate_ids) if sample_num > len(candidate_ids) else sample_num

            return random.sample(candidate_ids, sample_num)


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
        rel_neg_from = {}
        for from_id, r_ids in rel_from.items():
            rel_neg_from[from_id] = sample_negtive_ids([e[0] for e in entities], from_id, r_ids, ratio)

        # negtive from_id positive to_id
        rel_neg_to = {}
        for to_id, r_ids in rel_to.items():
            rel_neg_to[to_id] = sample_negtive_ids([e[0] for e in entities], to_id, r_ids, ratio)

        # 合并数据集
        for from_id, to_ids in rel_neg_from.items():
            for to_id in to_ids:
                relations.append((from_id, to_id, 0))

        for to_id, from_ids in rel_neg_to.items():
            for from_id in from_ids:
                relations.append((from_id, to_id, 0))

        relations = list(set(relations))
        return relations


    def gen_re_data(self, text, entities, relations):
        """
        根据关系对数据生成训练数据：
        entities: [(id, start, end, text, label)]
        relations: [(from_id, to_id, label)]
        在from_id前后加'#'，在to_id前后加'$'，并加上from_id、to_id开始结束位置以及关系标签
        """
        entities_dict = {e[0]:e for e in entities}

        results = []
        for from_id, to_id, label in relations:
            # 标记开始结束位置
            from_start_pos = entities_dict[from_id][1]
            from_end_pos = entities_dict[from_id][2]
            to_start_pos = entities_dict[to_id][1]
            to_end_pos = entities_dict[to_id][2]

            if from_start_pos < to_start_pos:
                item_text = text[:from_start_pos] + '#' + text[from_start_pos:from_end_pos] + '#' + text[from_end_pos:to_start_pos] + '$' + text[to_start_pos:to_end_pos] + '$' + text[to_end_pos:]
                results.append('%s\t%s\t%s\t%s\t%s\t%s' % (item_text, from_start_pos, from_end_pos+1, to_start_pos+2, to_end_pos+3, label))
            else:
                item_text = text[:to_start_pos] + '$' + text[to_start_pos:to_end_pos] + '$' + text[to_end_pos:from_start_pos] + '#' + text[from_start_pos:from_end_pos] + '#' + text[from_end_pos:]
                results.append('%s\t%s\t%s\t%s\t%s\t%s' % (item_text, from_start_pos+2, from_end_pos+3, to_start_pos, to_end_pos+1, label))

        return results


    def process_gene_ner(self, input, balance_strategy):
        """
        生成Gene表达NER训练数据
        """
        json_data = self.ls_transfomer.load_json_file(input)
        results = []
        for item in json_data:
            for item1 in self.split_item_by_sep(item, separator='\n', append_sep=False):
                for item2 in self.split_item_by_sep(item1, separator='. ', append_sep=True):
                    entities = self.ls_transfomer.get_entities(item2)
                    r = self.gen_ner_data(item2["data"]["text"], entities, language='EN')
                    if r is not None:
                        results.append(r)

        # self.write_ner_data(self.balance_ner_data(results, strategy=balance_strategy), output)
        return self.balance_ner_data(results, strategy=balance_strategy)

    def process_gene_ner_forspacy(self, input, balance_strategy):
        """
        生成Gene表达NER训练数据，for spacy
        """
        json_data = self.ls_transfomer.load_json_file(input)
        results = []
        for item in json_data:
            for item1 in self.split_item_by_sep(item, separator='\n', append_sep=False):
                for item2 in self.split_item_by_sep(item1, separator='. ', append_sep=True):
                    entities = self.ls_transfomer.merge_entities(self.ls_transfomer.get_entities(item2))
                    if len(entities) > 0:
                        results.append((item2["data"]["text"], [(e[1], e[2], e[4]) for e in entities]))


        # self.write_ner_data(self.balance_ner_data(results, strategy=balance_strategy), output)
        return self.balance_ner_data_forspacy(results, strategy=balance_strategy)


    def process_clinic_ner(self, input, output):
        """
        生成疾病诊断NER训练数据
        """
        json_data = self.ls_transfomer.load_json_file(input)
        results = []
        for item in json_data:
            for item1 in self.split_item_by_sep(item, separator='\n', append_sep=False):
                for item2 in self.split_item_banlance_txtlength(item2, delta=1000):
                    entities = self.ls_transfomer.get_entities(item2)
                    r = self.gen_ner_data(item2["data"]["text"], entities, ['症状', '部位', '性质'])
                    if r is not None:
                        results.append(r)

        self.write_ner_data(self.balance_ner_data(results), output)


    def process_gene_re(self, input, output, type='train', cls_num=250):
        all_entities = []
        for item in self.ls_transfomer.load_json_file(input):
            for item1 in self.split_item_by_sep(item, separator='\n', append_sep=False):
                for item2 in self.split_item_by_sep(item1, separator='. ', append_sep=True):
                    all_entities.extend(self.ls_transfomer.get_entities(item2))

        lt_dict = self.ls_transfomer.get_entity_label_texts(all_entities)

        # 生成关系标记数据
        cls_texts_dict, o_texts = {'null': []}, []
        for item in self.ls_transfomer.load_json_file(input):
            # 判断是否跨句关系
            # rl_ct = len(self.ls_transfomer.get_relations(item))
            # rl_ct_split = 0
            for item1 in self.split_item_by_sep(item, separator='\n', append_sep=False):
                # 没有标记的数据过滤掉
                # if len(item1["annotations"][0]['result']) == 0:
                #     continue
                for item2 in self.split_item_by_sep(item1, separator='. ', append_sep=True):
                    # NER标记<2的数据过滤掉
                    entities = self.ls_transfomer.get_entities(item2)
                    text = item2["data"]["text"].strip()
                    if len(text) < 20:
                        continue
                    # if len(entities) < 2:
                    #     if len(text) > 50:
                    #         # o_texts.append(self.ls_transfomer.replace_entity_by_label(text, entities))
                    #         o_texts.append((text, entities))
                    #     continue

                    relations = self.ls_transfomer.get_relations(item2)
                    if len(relations) == 0:
                        # cls_texts_dict['null'].append(self.ls_transfomer.replace_entity_by_label(text, entities))
                        cls_texts_dict['null'].append((text, entities))
                    else:
                        # rl_ct_split = rl_ct_split + len(relations)
                        # relations = self.expand_neg_relations(entities, relations, ratio)
                        # results.extend(self.gen_re_data(text, entities, relations))
                        # results.append('%s\t0\t0\t0\t0\t%s' % (text, 1))
                        for from_id, to_id, label in relations:
                            r_entities = self.ls_transfomer.search_entities_in_list(entities, [from_id, to_id])
                            if label not in cls_texts_dict:
                                 cls_texts_dict[label] = []
                            # cls_texts_dict[label].append(self.ls_transfomer.replace_entity_by_label(text, r_entities))
                            cls_texts_dict[label].append((text, r_entities))

            # if rl_ct != rl_ct_split:
            #     print(item)

        if type == 'train':
            # 数据增强
            results = self.data_augment_rp_entity(cls_texts_dict, lt_dict, cls_num)
            random.shuffle(results)
        else:
            results = []
            for r_label, vals in cls_texts_dict.items():
                for text, entities in vals:
                    results.append((self.replace_entity_by_add_simbol(text, entities), r_label))

        self.write_lines(results, output)


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
    parser.add_argument('-i', type=str, default='project.json', help='input file')
    parser.add_argument('-o', type=str, default='train_data.txt', help='output file')
    parser.add_argument('-t', type=str, default='NER', help='data type')
    parser.add_argument('-r', type=float, default=0.5, help='negtive sample ratio')
    parser.add_argument('-l', type=str, default='CH', help='language')
    parser.add_argument('-p', type=str, default='gene', help='project') # [gene, clinic]
    args = parser.parse_args()

    input = args.i
    output = args.o
    type = args.t
    ratio = args.r
    language = args.l
    project = args.p

    print("input: %s, output: %s, runtype: %s, project: %s, language: %s, ratio: %s" % (input, output, type, project, language, ratio))
    if type not in ['NER', 'RE']:
        print('Error: parameter type must be one of [NER, RE]')
        exit()

    gen = TrainDataGenerator()
    if project == 'gene':
        if type == 'RE':
            gen.process_gene_re(input, output, ratio)
        else:
            gen.process_gene_ner(input, output)
    elif project == 'clinic':
        if type == 'RE':
            gen.process_clinic_re(input, output)
        else:
            gen.process_clinic_ner(input, output)

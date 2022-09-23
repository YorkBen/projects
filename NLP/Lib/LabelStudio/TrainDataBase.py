import random
import copy

class TrainDataBase:
    def balance_data(self, data_dict, strategy='min'):
        """
        均衡数据。
        data_dict：{key: [data_line]}
        """
        results = []

        print('balance data...')
        print('strategy: %s' % strategy)
        # balance策略，选最小的
        len_dict = {label:len(data) for label, data in data_dict.items()}
        print('data length dict: ', len_dict)
        len_arr = [len(data) for label, data in data_dict.items()]

        if strategy == 'min':
            choose_nums = [min(len_arr) for l in len_arr]
        elif strategy == 'max':
            choose_nums = [max(len_arr) for l in len_arr]
        else:
            choose_nums = len_arr
        print('choose nums: ', choose_nums)
        for choose_num, (label, data) in zip(choose_nums, data_dict.items()):
            label_data = copy.copy(data)
            while len(label_data) < choose_num:
                label_data.extend(data)
            results.extend(random.sample(label_data, choose_num))

        random.shuffle(results)

        return results


    def data_augment_ap_text(self, data_dict, augment_texts, cls_num):
        """
        数据增强。每类和补充文本中各选一个合成语句，每类返回cls_num个数据。
        """
        augment_results = []
        for k, vals in data_dict.items():
            gen_num = cls_num if k == 'null' else cls_num
            for i in range(0, gen_num):
                s0, s_aug = random.choice(vals), random.choice(augment_texts)
                if len(s0) + len(s_aug) >= 500:
                    augment_results.append(('%s %s' % (s0, s_aug), k))
                else:
                    if random.randint(0, 1) == 0:
                        augment_results.append(('%s %s' % (s0, s_aug), k))
                    else:
                        augment_results.append(('%s %s' % (s_aug, s0), k))

        for i in range(0, cls_num):
            augment_results.append(('%s %s' % (random.choice(augment_texts), random.choice(augment_texts)), '0'))

        return augment_results


    def replace_entity_by_other_entities(self, text, entities, lt_dict):
        """
        替换文本中的实体字符串为实体标签。
        entities是排过序的
        lt_dict是标签到文字数组的字典
        """
        r_text = ''
        a_start = 0
        for _, start, end, _, label in entities:
            r_text = r_text + text[a_start:start]
            r_text = r_text + '#%s#' % random.choice(lt_dict[label])
            # r_text = r_text + '%s' % random.choice(lt_dict[label])
            a_start = end
        r_text = r_text + text[a_start:]
        return r_text

    def replace_entity_by_label(self, text, entities):
        """
        替换文本中的实体字符串为实体标签。
        entities是排过序的
        """
        entities = sorted(entities, key=lambda x: x[1])
        r_text = ''
        a_start = 0
        for _, start, end, _, label in entities:
            r_text = r_text + text[a_start:start]
            r_text = r_text + '#%s#' % label
            a_start = end
        r_text = r_text + text[a_start:]

        return r_text

    def replace_entity_by_add_simbol(self, text, entities):
        """
        文本中实体字符串前后加上#
        """
        entities = sorted(entities, key=lambda x: x[1])
        r_text = ''
        a_start = 0
        for _, start, end, e_txt, label in entities:
            r_text = r_text + text[a_start:start]
            r_text = r_text + '#%s#' % e_txt
            # r_text = r_text + '%s' % e_txt
            a_start = end
        r_text = r_text + text[a_start:]

        return r_text


    def data_augment_rp_entity(self, data_dict, entity_label_txts, cls_num):
        """
        数据增强。使用同类实体随机替换原实体。
        data_dict: {relation_label: [(text, entities)]}
        entity_label_txts: {entity_label: txt_arr}
        """
        augment_results = []
        for r_label, vals in data_dict.items():
            r_label_txts = []
            # 原始文本
            for text, entities in vals:
                r_label_txts.append((self.replace_entity_by_add_simbol(text, entities), r_label))

            # 替换为标签
            for text, entities in vals:
                r_label_txts.append((self.replace_entity_by_label(text, entities), r_label))

            # 替换实体
            while len(r_label_txts) < cls_num:
                for text, entities in vals:
                    r_label_txts.append((self.replace_entity_by_other_entities(text, entities, entity_label_txts), r_label))

            augment_results.extend(r_label_txts[:cls_num])

        random.shuffle(augment_results)

        return augment_results


    def write_lines(self, data, file_path):
        """
        将数据行写入文件
        """
        with open(file_path, 'w') as f:
            for l in data:
                if isinstance(l, tuple) or isinstance(l, list):
                    s = '\t'.join(l)
                else:
                    s = l
                f.write('%s\n' % s)

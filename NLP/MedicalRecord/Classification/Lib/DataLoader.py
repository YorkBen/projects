import random
import numpy as np
import pandas as pd
import logging

class DataLoader:
    def load_data_lines(self, file_path, separate=True, num_fields=2, separator=',', skip_title=True, shuffle=True):
        """
        把文件行加载到数组，去掉首尾空格。
        separate: 是否拆分文本行
        num_fields：每行包含的数据个数，如果为None表示不需要验证
        separator：行拆分分隔符
        skip_title：是否省去第一行
        shuffle：是否打乱顺序
        """
        logging.debug('Loading data lines...')
        lines = []
        start_line = 1 if skip_title else 0
        with open(file_path, "r") as f:
            for line in f.readlines()[start_line:]:
                # 如果要拆分行
                if separate:
                    arr = line.strip().split(separator)
                    if len(arr) == num_fields:
                        lines.append(arr)
                    else:
                        logging.warning('Skipping Illegal Line: %s!' % line)
                else:
                    lines.append(line.strip())

        if shuffle:
            logging.debug('Shuffle data lines...')
            random.shuffle(lines)

        logging.debug('%s Lines Loaded!' % len(lines))
        return lines


    def stat_cls_num(self, lines, cls_col):
        """
        统计数据各个类别的数量
        lines：数据数组
        cls_col：类别列
        """
        logging.debug('Statistic Class Sample Number...')
        cls_num_dict = {}
        for line in lines:
            elem = line[cls_col]
            if elem not in cls_num_dict:
                cls_num_dict[elem] = 1
            else:
                cls_num_dict[elem] = cls_num_dict[elem] + 1

        logging.debug('Class Sample Number: %s' % cls_num_dict)
        return cls_num_dict


    def balance_class_sample_num(self, lines, cls_col, cls_num_dict, stratege='medium'):
        """
        平衡类别样本数。
        lines：数据数组
        cls_num_dict：类别样本数量
        stratege：平衡策略。medium，按中位数来保留，大于的扔掉。minimum，按最少个数保留
        """
        logging.debug('Balancing Class Sample Number...')
        logging.debug('Balancing Stratege: %s' % stratege)
        vals = sorted(list(cls_num_dict.values()))
        if stratege == 'medium':
            # 获取中位数
            stratege_val = vals[len(vals) // 2]
        else:
            stratege_val = vals[0]
        logging.debug('Balancing Stratege Number: %s' % stratege_val)

        # 保留数据行
        np_lines = np.array(lines)
        results = None
        for key in cls_num_dict.keys():
            key_lines = np_lines[np_lines[:,cls_col] == key]
            if cls_num_dict[key] > stratege_val:
                to_concat = key_lines[:stratege_val]
            else:
                to_concat = key_lines

            if results is None:
                results = to_concat
            else:
                results = np.concatenate((results, to_concat),axis=0)

        logging.debug('%s Lines Keeped!' % len(results))
        return results.tolist()


    def load_text_labels(self, file_path, separator=',', skip_title=True, balance_stratege=None):
        """
        加载文本+标签数据
        """
        lines = self.load_data_lines(file_path=file_path, num_fields=2, separator=separator, skip_title=skip_title)
        cls_num_dict = self.stat_cls_num(lines, 1)
        if balance_stratege is not None:
            lines = self.balance_class_sample_num(lines, 1, cls_num_dict, stratege=balance_stratege)

        texts = [e[0] for e in lines]
        labels = [e[1] for e in lines]
        return texts, labels

    def load_text_pair_labels(self, file_path, separator=',', skip_title=True, balance_stratege=None):
        """
        加载文本+标签数据
        """
        lines = self.load_data_lines(file_path=file_path, num_fields=3, separator=separator, skip_title=skip_title)
        cls_num_dict = self.stat_cls_num(lines, 2)
        if balance_stratege is not None:
            lines = self.balance_class_sample_num(lines, 2, cls_num_dict, stratege=balance_stratege)

        texts_1 = [e[0] for e in lines]
        texts_2 = [e[1] for e in lines]
        labels = [e[2] for e in lines]
        return texts_1, texts_2, labels






#

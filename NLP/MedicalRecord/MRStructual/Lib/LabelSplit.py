import re
import json
import logging
import time
import sys

sys.path.append('../Lib')

from RegexUtil import RegexUtil

class LabelSplit:
    """
    使用Label文本拆分整段文本，并提取Label对应的文本值。
    Label之间没有次序关系，只要在label数组里，都是可以作为label
    1. 使用\n换行符来增强拆分逻辑
    2. 使用1、2、等序号符号增强拆分逻辑
    3. 没有label的文本也输出
    """
    def __init__(self):
        self.utils = RegexUtil()
        pass

    def split_str_by_matrix(self, str, matrix, keys):
        """
        """
        # 存放start_ind, score, key
        ordered_arr = []
        for key, row in zip(keys, matrix):
            row_ = sorted(row, key=lambda x: x[1], reverse=True)
            for elem in row_:
                if elem[1] >= 1.0 or elem[1] == -1:
                    ordered_arr.append(elem + (key,))
                    break

        ordered_arr = sorted(ordered_arr, key=lambda x: x[1], reverse=True)

        # 初始化切分点
        str_len = len(str)
        split_points = [0 for i in range(str_len)]
        split_keys = ['' for i in range(str_len)]
        for item in ordered_arr:
            split_points[item[0]] = item[1]
            split_keys[item[0]] = item[2]

        results = {key:'' for key in keys}
        for item in ordered_arr:
            if item[1] >= 1.0:
                start = item[0]
                key = split_keys[start]
                ind = start + 1
                while ind < str_len and split_points[ind] == 0:
                    split_points[ind] = -1
                    ind = ind + 1
                results[key] = str[start:ind]

        return results


    def process(self, str, keys):
        """
        使用动态规划，计算最大概率方法来拆分。
        """
        matrix = []
        for ind, key in enumerate(keys):
            matrix.append(self.utils.find_key_in_str_prob(key, str))

        for i in range(0, len(matrix)):
            logging.debug('%s----->%s----->%s' % (i, keys[i], self.utils.print_2d_arr(matrix[i])))

        results = self.split_str_by_matrix(str, matrix, keys)

        return results

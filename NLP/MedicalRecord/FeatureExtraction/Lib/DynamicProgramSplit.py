import re
import json
import logging
import time
from Lib.Utils import Utils

class DynamicProgramSplit:
    def __init__(self):
        """
        使用动态规划算法来找最优分割节点
        """
        self.matrix_max_cache = {}   # i^j -> max_value
        self.utils = Utils()


    def calc_max(self, i, j, matrix, max_pos):
        """
        动态规划算法，递归函数。
        递归计算i行j列及子节点路径的最大概率
        """
        key = '%s^%s' % (i, j)
        if key not in self.matrix_max_cache:
            # 最后一行
            if i == len(matrix) - 1:
                self.matrix_max_cache[key] = (i, j, matrix[i][j][1])
            else:
                max_pos = matrix[i][j][0] if matrix[i][j][0] != -1 else max_pos
                # 下一行的关键词匹配位置必须在当前匹配位置后面，或者下一行没有匹配
                max_arr = [(k, self.calc_max(i+1, k, matrix, max_pos)[2]) for k in range(0, len(matrix[i+1])) if matrix[i+1][k][0] > max_pos or matrix[i+1][k][0] == -1]
                if len(max_arr) == 0:
                    self.matrix_max_cache[key] = (i, j, matrix[i][j][1])
                else:
                    max_arr = sorted(max_arr, key=lambda x: x[1], reverse=True)
                    max_val = matrix[i][j][1] + max_arr[0][1]
                    self.matrix_max_cache[key] = (i+1, max_arr[0][0], max_val)

        # print(matrix_max_cache[key])
        return self.matrix_max_cache[key]


    def find_path(self, matrix, keys):
        """
        动态规划后，找最大概率路径
        """
        max_pos = -1
        path = []
        i, j = 0, 0
        while i < len(matrix):
            max_pos = matrix[i][j][0] if matrix[i][j][0] != -1 else max_pos
            i, j, score = self.calc_max(i, j, matrix, max_pos)
            # 到达最后行
            if len(path) > 0 and len(path) == len(keys):
                break
            # 从第二行开始
            if i > 0:
                path.append((keys[i-1], matrix[i][j]))

        return path


    def split_str(self, str, path):
        """
        根据拆分路径来切分字符串。
        """
        result = {}
        # 拆分
        for i, p in enumerate(path):
            if p == [1][0] == -1:
                result[p[0]] = ''
            else:
                end = ''
                for k in range(i+1, len(path)):
                    if path[k][1][0] != -1:
                        end = path[k][1][0]
                        break
                if end == '':
                    result[p[0]] = str[p[1][0]:]
                else:
                    result[p[0]] = str[p[1][0]:end]

        return result


    def process(self, str, keys):
        """
        使用动态规划，计算最大概率方法来拆分。
        """
        # matrix格式：每一行为一个关键词的匹配结果，每一列存放：（匹配开始位置，匹配度分值）
        matrix = [[(-1, 0)]]
        for ind, key in enumerate(keys):
            matrix.append(self.utils.find_key_in_str_prob(key, str))

        for i in range(1, len(matrix)):
            logging.debug('%s----->%s----->%s' % (i, keys[i-1], self.utils.print_2d_arr(matrix[i])))

        self.matrix_max_cache = {}
        self.calc_max(0, 0, matrix, -1)
        path = self.find_path(matrix, keys)

        logging.debug('最优路径：')
        logging.debug('%s' % self.utils.print_2d_arr(path))

        result = self.split_str(str, path)

        return result
#

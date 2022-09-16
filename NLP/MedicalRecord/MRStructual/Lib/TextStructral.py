import re
import json
import logging
import time
import sys

from Lib.DynamicProgramSplit import DynamicProgramSplit
from Lib.LabelSplit import LabelSplit

sys.path.append('../Lib')

from RegexUtil import RegexUtil

class TextStructral:
    """
    文本结构化类。
    提供需要结构化文本以及其格式模板，解析出模板格式的内容键值对。
    模板有多层结构，因此会递归解析。
    """
    def __init__(self):
        self.Utils = RegexUtil()
        # self.keys = keys
        # 当字符串中出现如下关键词时，说明是不要的内容，需要去掉。
        # self.remove_keys = [key + '：' for key in self.keys]
        self.remove_keys = ['输血史：', '民族：', '婚育史：', '记录医师签名：', '上述病史记录已征得陈述者认同。', '医师：', '医师签名：']


    def load_records(self, data_file):
        """
        加载文本数据，每一段文本数据前后以空行分隔。
        """
        records = []
        record = []
        with open(data_file) as f:
            for line in f.readlines():
                if line.strip() == '':
                    if len(record) > 0:
                        records.append(record)
                        record = []
                else:
                    record.append(line)
            records.append(record)

        logging.debug('load records: %s' % len(records))
        self.records = records

        return records

    def load_template(self, template_file):
        """
        加载正则匹配模板
        """
        template = None
        with open(template_file) as f:
            template = json.load(f, strict=False)
            logging.debug('load template: %s' % str(template))

        self.template = template
        return self.template


    def set_processor(self, processor_type="dynamic"):
        """
        设置处理算法
        """
        if processor_type == 'dynamic':
            self.processor = DynamicProgramSplit()
        elif processor_type == 'label':
            self.processor = LabelSplit()


    def write_result(self, file_path, data=None):
        """
        将Json数据写入文件
        """
        with open(file_path, "w") as f:
            if data is None:
                f.write(json.dumps(self.results, indent=1, separators=(',', ':'), ensure_ascii=False))
            else:
                f.write(json.dumps(data, indent=1, separators=(',', ':'), ensure_ascii=False))


    def post_process_txt(self, val):
        # 去除病历中的[***]内容
        val = self.Utils.remove_squarebracket_cnt(val)
        # 去掉包含在移除键数组中的key
        val = self.Utils.remove_after_keys(val, self.remove_keys)
        # 去掉末尾的（
        val = re.sub(r'（\s*\n?$', '', val)
        # 去掉一个（
        val = re.sub(r'^\s*）\s*\n?$', '', val)
        # 去掉末尾的序号1、2、等
        val = re.sub(r'[1-9]+[、. ]+$', '', val)

        return val


    def process_1_layer(self, str, template):
        """
        递归函数，对所有的key->cnt，处理cnt的val识别
        str: 匹配文本
        template: 匹配模板
        """
        keys = list(template.keys())
        result = template.copy()
        # record_split = split_cnt_by_keys(str, keys)
        record_split = self.processor.process(str, keys)

        # 拆分内容
        for key in keys:
            str = record_split[key]
            if isinstance(template[key], dict):
                result[key] = self.process_1_layer(str, template[key])
            else:
                if key == 'DATE':
                    result[key] = self.Utils.format_date(str)
                else:
                    val = self.Utils.find_key_value_pattern(str, key)
                    val = self.post_process_txt(val)
                    result[key] = self.Utils.format_by_type(val, template[key]).strip()

        return result


    def process(self, num=-1):
        """
        主函数
        """
        self.results = []
        for idx, record in enumerate(self.records):
            if num == -1 or num > idx:
                # str = ''.join(record)
                str = self.Utils.remove_squarebracket_cnt(''.join(record))
                self.results.append(self.process_1_layer(str, self.template))

        return self.results


    def process_record(self, str):
        """
        处理单条记录
        """
        str = self.Utils.remove_squarebracket_cnt(str)
        return self.process_1_layer(str, self.template)
#

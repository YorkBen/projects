import time
import logging
import json
import sys

sys.path.append('../Lib')

from Lib.TextStructral import TextStructral
from FileUtil import load_file, load_dict
from RegexUtil import RegexUtil
from MRRecordUtil import process_mr

utils = RegexUtil()

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    ts = TextStructral()

    records = ts.load_records('data/records/测试.txt')
    ts.load_template('data/template/首次病程.json')
    ts.set_processor()
    results = ts.process()

    print(json.dumps(results, indent=1, separators=(',', ':'), ensure_ascii=False))

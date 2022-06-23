import time
import logging
import json
from Lib.TextStructral import TextStructral
from Lib.FileUtil import load_file
from Lib.Utils import Utils
from process_inspect import load_dict as load_insp_dict

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    ts = TextStructral()

    records = ts.load_records('data/records/测试.txt')
    ts.load_template('data/template/首次病程.json')
    ts.set_processor()
    results = ts.process()

    print(json.dumps(results, indent=1, separators=(',', ':'), ensure_ascii=False))

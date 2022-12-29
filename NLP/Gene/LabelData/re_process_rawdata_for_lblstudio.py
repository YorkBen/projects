import os
import re
import json
import random
# import threading
from multiprocessing import Process, Manager, Lock
import time
from genLabeledData import *

data_dir=r'/mnt/d/项目资料/基因表达/通路分析'
file_gene_path=os.path.join(data_dir, 'gene_alias.txt')
file_cancer_path=os.path.join(data_dir, '癌症.txt')
file_wnt4_path=os.path.join(data_dir, '(cancer) AND (WNT4)文献.txt')
file_ca_path = os.path.join(data_dir, '(cancer) AND (Carbonic Anhydrase 1)文献.txt')
file_ner_data = r'Data/ner_texts.txt'

def read_gene_data_file(file_path):
    data = []
    whole_line = ''
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line[:-1]
            whole_line = whole_line + line
            if not line.endswith(' '):
                data.append(whole_line)
                whole_line = ''
        data.append(whole_line)
    return data

def sel_item(gene_data):
    result, item = [], []
    start_nbr, i = 1, 0
    while i < len(gene_data):
        if gene_data[i].startswith('%s.' % start_nbr):
#             print('start nbr: %s match!' % start_nbr)
            if len(item) > 0:
                result.append(item)
                item = []

            start_nbr = start_nbr + 1
        item.append(gene_data[i])
        i = i + 1

    if len(item) > 0:
        result.append(item)

    print(len(result))
    return result

def proc_item(item_arr):
    results = []
    for item_idx, item in enumerate(item_arr):
        starts_search_flag = 'Author information:'
        ends_search_flag = '(^DOI:)|(^©)|(^PMID: )|(^Copyright)'
        # 标题
        result_item = [item[2]]
        for idx, line_str in enumerate(item):
            # 找Author Information
            if re.search(starts_search_flag, line_str):
                # 找空行
                no = idx + 1
                while not item[no] == '':
                    no = no + 1
                no = no + 1

                # 摘要前可能有Comment in
                if item[no].startswith('Comment in') or item[no].startswith('Erratum in'):
                    while not item[no] == '':
                        no = no + 1
                    no = no + 1

                # 找摘要
                while not item[no] == '':
                    result_item.append(item[no])
                    no = no + 1

                results.append(result_item)
                break

            # 找到结束记号
            if re.search(ends_search_flag, line_str):
                # 找空行
                ends_no = idx - 1
                starts_no = idx - 2
                while not item[starts_no] == '':
                    starts_no = starts_no - 1

                # 写内容
                result_item.extend(item[starts_no + 1:ends_no])

                results.append(result_item)
                break


    return results

def gen_train_val_data():
    data_wnt4 = sel_item(read_gene_data_file(file_wnt4_path))
    data_ca = sel_item(read_gene_data_file(file_ca_path))

    data_all = data_wnt4 + data_ca
    random.seed(2345)
    random.shuffle(data_all)
    train_len = int(len(data_all) * 0.8)
    train_data, test_data = data_all[:train_len], data_all[train_len:]

    with open('train_data.text', 'w') as f:
        for item in train_data:
            f.write('%s\n' % '\\n'.join(item))

    with open('test_data.text', 'w') as f:
        for item in test_data:
            f.write('%s\n' % '\\n'.join(item))


def load_gene_dict():
    gene_label_dict = {}
    with open(file_gene_path, encoding='utf-8') as f:
        for line in f.readlines():
            # RNA 不处理
            if 'Novel Transcript' in line:
                continue

            arr = line.strip().split('	')[1:]
            for e in arr:
                e = e.strip()
                if len(e) >= 2:
                    gene_label_dict[e] = 'Gene'

    return gene_label_dict

def load_cancel_dict():
    gene_label_dict = {}
    with open(file_cancer_path) as f:
        for line in f.readlines():
            e = line.strip()
            gene_label_dict[e] = 'Cancel'

    return gene_label_dict

def load_dict_from_ner_data():
    gene_label_dict = {}
    with open(file_ner_data) as f:
        for line in f.readlines():
            arr = line.strip().split('	')
            if len(arr) == 2:
                gene_label_dict[arr[0]] = arr[1]

    return gene_label_dict
    #
    # json_data = ''
    # with open(file_ner_data) as f:
    #     json_data = json.load(f, strict=False)
    #     # print(json.dumps(json_data[0], indent=1, separators=(',', ':'), ensure_ascii=False))
    #
    # # ner
    # ner_dict = {}
    # for item in json_data:
    #     for r in item['annotations'][0]['result']:
    #         if r["type"] == "labels":
    #             ner_dict[r['value']['text'].strip()] = r['value']['labels'][0]
    #
    # return ner_dict


def thread_run(text_data, idx):
    print("thread run: %s" % idx)
    time_start = time()
    json_data = process_by_regex(gene_dict, text_data, language='EN')
    results[idx] = json_data[0]
    time_end = time()
    print('time cost: %s' % (time_end - time_start))


if __name__ == "__main__":
    # gen_train_val_data()
    # gene_dict = load_gene_dict()
    gene_dict = {}
    gene_dict_1 = load_cancel_dict()
    gene_dict_2 = load_dict_from_ner_data()
    gene_dict |= gene_dict_1
    gene_dict |= gene_dict_2

    # print('B' in gene_dict)
    # print('B' in gene_dict_2)
    # # print(json.dumps(gene_dict_2, indent=1, separators=(',', ':'), ensure_ascii=False))
    # exit()

    thread_max_num = 14
    random.seed(2345)

    inhibits_data = []
    with open('../Data/20221207/inhibits_data.text') as f:
        for item in f.readlines():
            inhibits_data.append(item[:-1].replace('\\n', '\n'))

    promotes_data = []
    with open('../Data/20221207/promotes_data.text') as f:
        for item in f.readlines():
            promotes_data.append(item[:-1].replace('\\n', '\n'))

    uncertain_data = []
    with open('../Data/20221207/uncertain_data.text') as f:
        for item in f.readlines():
            uncertain_data.append(item[:-1].replace('\\n', '\n'))

    random.shuffle(inhibits_data)
    random.shuffle(promotes_data)
    random.shuffle(uncertain_data)

    uncertain_data_len_half = len(uncertain_data) // 2

    inhibits_data.extend(uncertain_data[:uncertain_data_len_half])
    promotes_data.extend(uncertain_data[uncertain_data_len_half:])
    random.shuffle(inhibits_data)
    random.shuffle(promotes_data)

    for out_path, text_data in zip(['../Data/20221207/promotes_data.json', '../Data/20221207/inhibits_data.json'], [promotes_data, inhibits_data]):
        print('processing %s' % out_path)
        text_data_len = len(text_data)
        results = Manager().list()
        for i in range(text_data_len):
            results.append(None)

        for k1 in range(0, text_data_len, thread_max_num):
            threads, thread_idxs = [], []
            for k2 in range(0, thread_max_num):
                idx = k1 + k2
                if idx < text_data_len:
                    # t = threading.Thread(target=thread_run, args=(text_data[idx], idx))
                    t = Process(target=thread_run, args=([text_data[idx]], idx))
                    threads.append(t)
                    thread_idxs.append(idx)

            # 开启进程
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # # 检测这轮线程是否都执行完
            # for idx in thread_idxs:
            #     if results[idx] is None:
            #         time.sleep(5)

            # 本轮线程都执行完，先写结果
            print('write file...')
            with open(out_path, "w") as f:
                f.write(json.dumps([item for item in results if item is not None], indent=1, separators=(',', ':'), ensure_ascii=False))

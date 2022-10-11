import json
import argparse

in_files = ['训练集_现病史_模型.txt', '训练集_既往史_模型.txt']
out_file = 'train_mdl.jsonl'

# in_files = ['测试集_现病史_模型.txt', '测试集_既往史_模型.txt']
# out_file = 'dev_mdl.jsonl'

# in_files = ['人机_现病史_模型.txt', '人机_既往史_模型.txt']
# out_file = 'test_mdl.jsonl'

if __name__ == '__main__':
    # # 参数
    # parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    # parser.add_argument('-i', type=str, default='2409', help='postfix num')
    # parser.add_argument('-o', type=str, default='腹痛', help='数据类型')
    # args = parser.parse_args()
    #
    # postfix = args.p
    # data_type = args.t

    id = 0
    data = []
    for in_file in in_files:
        cols = []
        with open(in_file) as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    cols = line.strip().split('	')
                else:
                    arr = line.strip().split('	')
                    text = arr[2]
                    for col_idx in range(3, len(arr)):
                        col_lbl = '未知'
                        if arr[col_idx] == '0':
                            col_lbl = '阴性'
                        elif arr[col_idx] == '1':
                            col_lbl = '阳性'
                        data.append({
                            "id": id+1,
                            "text": text,
                            "label": [col_lbl],
                            "prompt_prefix": cols[col_idx]
                        })


    with open(out_file, "w", encoding="utf-8") as f:
        count = 0
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    print("Save %d examples to %s." % (count, out_file))

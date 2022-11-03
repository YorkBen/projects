import json
import argparse

# in_files, out_file = ['训练_全特征_多诊断.txt'], 'train_mdl_num.jsonl'

# in_files, out_file = ['测试_全特征_多诊断.txt'], 'dev_mdl_num.jsonl'

# in_files = ['人机_现病史_模型.txt', '人机_既往史_模型.txt']
# out_file = 'test_mdl.jsonl'
labels = ["急性阑尾炎", "急性胰腺炎", "肠梗阻", "异位妊娠", "急性胆管炎", "急性胆囊炎", "上尿路结石", "卵巢囊肿", "消化道穿孔"]

if __name__ == '__main__':
    # # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-i', type=str, default='训练_全特征_多诊断.txt', help='输入数据')
    parser.add_argument('-o', type=str, default='train_mdl_num.jsonl', help='输出数据')
    parser.add_argument('-t', type=str, default='多诊断', help='类型：单诊断|多诊断')
    args = parser.parse_args()

    input = args.i
    output = args.o
    data_type = args.t

    feature_start_col = 11 if data_type == '多诊断' else 3

    id = 0
    data = []
    for in_file in [input]:
        cols = []
        with open(in_file) as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    cols = line.strip().split('	')
                else:
                    arr = line.strip().split('	')
                    text_pos_arr, text_uk_arr, lbl_arr = [], [], []
                    text = arr[2]

                    # labels
                    if data_type == '多诊断':
                        for col_idx in range(2, feature_start_col):
                            if arr[col_idx] == '1':
                                lbl_arr.append(cols[col_idx])
                    else:
                        col_idx = int(arr[2])
                        lbl_arr.append(labels[col_idx])

                    # ### text
                    # for col_idx in range(feature_start_col, len(arr)):
                    #     if arr[col_idx] == '1':
                    #         text_pos_arr.append(cols[col_idx])
                    #     elif arr[col_idx] not in ['0', '1', '2']:
                    #         text_pos_arr.append(cols[col_idx] + arr[col_idx])
                    #
                    # data.append({
                    #     "id": id+1,
                    #     "text": '，'.join(text_pos_arr),
                    #     "label": lbl_arr,
                    #     "prompt_prefix": '预测疾病'
                    # })

                    # num text
                    data.append({
                        "id": id+1,
                        "text": '，'.join(arr[feature_start_col:]),
                        "label": lbl_arr,
                        "prompt_prefix": '预测疾病'
                    })


    with open(output, "w", encoding="utf-8") as f:
        count = 0
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    print("Save %d examples to %s." % (count, output))

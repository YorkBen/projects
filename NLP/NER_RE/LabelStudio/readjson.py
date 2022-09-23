import json
import argparse


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
    parser.add_argument('-i', type=str, default='project.json', help='input file')
    parser.add_argument('-n', type=int, default=1, help='output lines')
    args = parser.parse_args()

    input = args.i
    num = args.n

    print("input: %s, num: %s" % (input, num))

    # 加载json数据
    json_data = ''
    with open(input) as f:
        json_data = json.load(f, strict=False)
        for i in range(num):
            print(json_data[i])

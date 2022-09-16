import os
import argparse

from Lib.LabelStudioTransformer import Transformer


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-i1', type=str, default='project.json', help='project.json')
    parser.add_argument('-i2', type=str, default='manual.json', help='manual.json')
    parser.add_argument('-o', type=str, default='merged.json', help='merged.json')
    args = parser.parse_args()

    input_p = args.i1
    input_m = args.i2
    output = args.o

    if not os.path.exists(input_p):
        print('%s not exists' % input_p)
        exit()
    if not os.path.exists(input_m):
        print('%s not exists' % input_m)
        exit()

    t = Transformer()
    t.write_json_file(t.merge_project_and_manual_data(input_p, input_m), output)

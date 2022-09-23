import sys
import argparse

sys.path.append('../../Lib/LabelStudio')

from TrainDataGenerator import TrainDataGenerator


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
    parser.add_argument('-i', type=str, default='project.json', help='input file')
    parser.add_argument('-o', type=str, default='train_data.txt', help='output file')
    parser.add_argument('-rt', type=str, default='NER', help='run type')
    parser.add_argument('-dt', type=str, default='train', help='data type')
    parser.add_argument('-bs', type=str, default='none', help='balance strategy')
    args = parser.parse_args()

    input = args.i
    output = args.o
    run_type = args.rt
    data_type = args.dt
    balance_strategy = args.bs

    print("input: %s, output: %s, runtype: %s, datatype: %s" % (input, output, run_type, data_type))
    if run_type not in ['NER', 'RE']:
        print('Error: parameter rt must be one of [NER, RE]')
        exit()
    if data_type not in ['train', 'test']:
        print('Error: parameter dt must be one of [NER, RE]')
        exit()

    gen = TrainDataGenerator()
    if run_type == 'RE':
        # gen.process_gene_re(input, output, type='train', cls_num=250)
        gen.process_gene_re(input, output, type=data_type)
    else:
        gen.process_gene_ner(input, output, balance_strategy=balance_strategy)

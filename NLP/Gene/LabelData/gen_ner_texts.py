import sys
import argparse

sys.path.append('../../Lib/LabelStudio')

from LabelStudioTransformer import Transformer

if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
    parser.add_argument('-i', type=str, default='project.json', help='input file')
    parser.add_argument('-o', type=str, default='train_data.txt', help='output file')
    args = parser.parse_args()

    input = args.i
    output = args.o

    print("input: %s, output: %s" % (input, output))

    t = Transformer()
    results = []
    for item in t.load_json_file(input):
        for e in t.get_entities(item):
            results.append((e[3], e[4]))
    results = list(set(results))
    results = sorted(results, key=lambda x: x[1])

    with open(output, 'w') as f:
        for e in results:
            f.write('%s\t%s\n' % (e[0], e[1]))

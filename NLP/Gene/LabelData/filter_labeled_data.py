import os
import argparse
import sys
import random

sys.path.append('../../Lib/LabelStudio')

from TrainDataGenerator import TrainDataGenerator
from LabelStudioTransformer import Transformer

"""从基因表达标记数据中筛选出指定的NER和RE的数据"""

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-i', type=str, default='project.json', help='project.json')
    parser.add_argument('-o', type=str, default='out.json', help='out.json')
    args = parser.parse_args()

    input = args.i
    output = args.o

    if not os.path.exists(input):
        print('%s not exists' % input)
        exit()

    dg = TrainDataGenerator()
    ls_transfomer = Transformer()

    json_data = ls_transfomer.load_json_file(input)
    results = []

    for item in json_data:
        for item1 in dg.split_item_by_sep(item, separator='\n', append_sep=False):
            for item2 in dg.split_item_by_sep(item1, separator='. ', append_sep=True):

                entities = ls_transfomer.get_entities(item2)
                entities_ = []
                has_special_entity = False
                for e in entities:
                    if e[-1] in ["Gene", "MultiGene", "Cancer", "MultiCancer", "signal pathway"]:
                        label = "Gene"
                        if e[-1] == "MultiGene":
                            label = "Multiply Gene"
                            has_special_entity = True
                        elif e[-1] == "Cancer":
                            label = "Cancer"
                            has_special_entity = True
                        elif e[-1] == "MultiCancer":
                            label = "Multiply Cancer"
                            has_special_entity = True
                        elif e[-1] == "signal pathway":
                            label = "Signal Pathway"
                            has_special_entity = True

                        entities_.append((e[0], e[1], e[2], e[3], label))
                entity_anns = [ls_transfomer.assemble_ner_entity_ann(*e) for e in entities_]

                relations = ls_transfomer.get_relations(item2)
                relations_ = []
                for r in relations:
                    if r[-1] in ["gene:positive", "gene:negtive", "gene:relatied", "gene:positive relatied", "gene:negtive relatied"]:
                        label = 'promote'
                        if r[-1] == "gene:positive":
                            label = 'promote'
                        elif r[-1] == "gene:negtive":
                            label = 'inhibit'
                        elif r[-1] == "gene:relatied":
                            label = 'is relatied to'
                        elif r[-1] == "gene:positive relatied":
                            label = 'is positivly relatied to'
                        elif r[-1] == "gene:negtive relatied":
                            label = 'is negativly relatied to'

                        relations_.append((r[0], r[1], label))
                rel_anns = [ls_transfomer.assembel_relation_entity_ann(*r) for r in relations_]

                if len(relations_) > 0 or has_special_entity:
                    anns = entity_anns + rel_anns
                    # random.shuffle(anns)
                    results.append(ls_transfomer.assemble_anns(item2["data"]["text"], anns))
                    # results.append(ls_transfomer.assemble_anns(item2["data"]["text"], entity_anns))


    ls_transfomer.write_json_file(results, output)

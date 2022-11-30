import os
import argparse
import sys
import random

sys.path.append('../../Lib/LabelStudio')

from TrainDataGenerator import TrainDataGenerator
from LabelStudioTransformer import Transformer

"""从基因表达标记数据中筛选出指定的NER和RE的数据"""

entity_map = {
    "Gene": "Gene",
    "MultiGene": "Multiply Gene",
    "Cancer": "Cancer",
    "MultiCancer": "Multiply Cancer",
    "signal pathway": "Signal Pathway",
    "GeneFunction": "Gene Function",
    "Gene multiFunction": "Gene MultiFunction"
}

relation_map = {
    "gene:positive": 'promotes',
    "gene:negtive": 'inhibits',
    "gene:relatied": 'is related to',
    "gene:promote dependence": "promotes the dependent gene of",
    "gene:inhibite dependence": "inhibits the dependent gene of",
    "gene:promote target": "promotes the target gene of",
    "gene:inhibite target": "inhibits the target gene of",
    "gene:dependence": "is dependent on",
    "gene:target": "'s target genes is",
    "gene:transcriptional coactivation": "is the transcriptional coactivation of",
    "gene:promote pathway": "promotes the signaling pathway of",
    "gene:inhibite pathway": "inhibits the signaling pathway of",
    "gene:pathway": "'s signaling pathway contains",
    "gene:responsive": "is the responsive gene of",
    "gene:positive relatied": "is positively related to",
    "gene:negtive relatied": "is negatively related to",
    "gene:upstream": "is upstream of",
    "gene:downstream": "is downstream of",
    "gene:Gene function": "act as the function of",
    "gene:Gene multifunction": "act as the multi-function of",

    "gene:abbrev": "is the abbreviation of"
}

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
                    label = entity_map[e[-1]]
                    if e[-1] in ["MultiGene", "Cancer", "MultiCancer", "signal pathway"]:
                        has_special_entity = True

                    entities_.append((e[0], e[1], e[2], e[3], label))
                entity_anns = [ls_transfomer.assemble_ner_entity_ann(*e) for e in entities_]

                relations = ls_transfomer.get_relations(item2)
                relations_ = []
                for r in relations:
                    if r[-1] not in ["gene:abbrev"]:
                        # if r[-1] == 'null':
                        #     label = 'null'
                        # else:
                        label = relation_map[r[-1]]
                        relations_.append((r[0], r[1], label))
                rel_anns = [ls_transfomer.assembel_relation_entity_ann(*r) for r in relations_]

                if len(relations_) > 0 or has_special_entity:
                    anns = entity_anns + rel_anns
                    # random.shuffle(anns)
                    results.append(ls_transfomer.assemble_anns(item2["data"]["text"], anns))
                    # results.append(ls_transfomer.assemble_anns(item2["data"]["text"], entity_anns))


    ls_transfomer.write_json_file(results, output)

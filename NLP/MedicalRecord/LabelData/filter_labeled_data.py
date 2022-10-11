import os
import argparse
import sys
import random

sys.path.append('../../Lib/LabelStudio')

from TrainDataGenerator import TrainDataGenerator
from LabelStudioTransformer import Transformer

"""从医学生标记的恶心呕吐数据中筛选出做NER和RE的数据"""

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-i', type=str, default='project.json', help='project.json')
    parser.add_argument('-o', type=str, default='out.json', help='merged.json')
    args = parser.parse_args()

    input = args.i
    output = args.o

    if not os.path.exists(input):
        print('%s not exists' % input)
        exit()

    dg = TrainDataGenerator()
    ls_transfomer = Transformer()

    symp_labels = ['畏寒', '恶心', '呕吐', '呕血', '嗳气', '反酸', '眩晕', '营养不良', '腹泻', '水肿', '抽搐', '高血糖']

    json_data = ls_transfomer.load_json_file(input)
    results = []
    for item in json_data:
        for item1 in dg.split_item_by_sep(item, separator='\n', append_sep=False):
            if item1["data"]["text"].startswith('生命体征:') or \
                item1["data"]["text"].startswith('CR') or \
                item1["data"]["text"].startswith('CT') or \
                item1["data"]["text"].startswith('US') or \
                item1["data"]["text"].startswith('DR') or \
                item1["data"]["text"].startswith('MR') or \
                item1["data"]["text"].startswith('X线'):
                continue

            for item2 in dg.split_item_banlance_txtlength(item1, delta=500):
                rl_entity_ids = []
                relations = ls_transfomer.get_relations(item2)
                for r in relations:
                    rl_entity_ids.append(r[0])
                    rl_entity_ids.append(r[1])

                entities = ls_transfomer.get_entities(item2)
                entities_sel = []
                for e in entities:
                    if e[0] in rl_entity_ids or \
                        e[-1] in symp_labels or \
                        e[-1] in ['症状']:
                        e_ = (e[0], e[1], e[2], e[3], e[-1] if e[-1] in ['否定词', '部位', '性质'] else '症状')
                        entities_sel.append(e_)

                if len(entities_sel) == 0:
                    continue
                # entities = ls_transfomer.merge_entities(entities_sel)
                entities = entities_sel
                entity_anns = [ls_transfomer.assemble_ner_entity_ann(*e) for e in entities]
                # print(entities)
                # print(relations)
                # print('')

                entity_label_dict = {e[0]:e[-1] for e in entities}
                relations_ = []
                for r in relations:
                    label1 = entity_label_dict[r[0]]
                    label2 = entity_label_dict[r[1]]
                    label = ''
                    if label1 == '症状' and label2 != '症状':
                        r0, r1 = r[0], r[1]
                        label = label2
                    if label2 == '症状' and label1 != '症状':
                        r0, r1 = r[1], r[0]
                        label = label1

                    if label == '否定词':
                        label = '否定'
                    elif label == '部位':
                        label = '部位'
                    elif label == '性质':
                        label = '性质'

                    if label != '':
                        relations_.append((r0, r1, label))
                rel_anns = [ls_transfomer.assembel_relation_entity_ann(*r) for r in relations_]

                anns = entity_anns + rel_anns
                # random.shuffle(anns)
                results.append(ls_transfomer.assemble_anns(item2["data"]["text"], anns))
                # results.append(ls_transfomer.assemble_anns(item2["data"]["text"], entity_anns))


    ls_transfomer.write_json_file(results, output)

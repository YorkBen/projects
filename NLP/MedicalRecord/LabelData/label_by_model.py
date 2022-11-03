"""
使用NER和关系标注模型对数据进行标注，并转换为LabelStudio格式
"""
import sys
import json
import argparse

sys.path.append('../../Lib/LabelStudio')
from LabelStudioTransformer import Transformer
transformer = Transformer()

sys.path.append('../Lib')
from MRRecordUtil import *
from RegexBase import RegexBase

from paddlenlp import Taskflow
from pprint import pprint

schema = {'症状': ['部位', '否定词', '性质']}
ie = Taskflow("information_extraction", schema=schema, task_path='../../paddlenlp/uie_zh/checkpoint_ext/model_best') #  device_id=-1

# text = '患者2018-10-19我院移植冻胚2枚，移植后10天查血HCG示：501.8IU/L，最近半月以来患者无明显诱因间断出现恶心，伴呕吐，为胃内容物，近3天患者感呕吐明显加重，患者为求进一步诊治，来我院就诊，今我院门诊查尿常规示:酮体3+，门诊以\"妊娠剧吐伴酮症\"收住院'
# pprint(ie(text))

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='Select Data With MR_NOs')
    parser.add_argument('-i', type=str, default='project.json', help='project.json')
    parser.add_argument('-o', type=str, default='out.json', help='merged.json')
    args = parser.parse_args()

    input = args.i
    output = args.o


    records = transformer.load_json_file(input)
    results = []
    for tid, record in enumerate(records):
        texts = get_json_value(record, ['入院记录', '主诉']) + '\n\n'
        texts = texts + get_json_value(record, ['入院记录', '现病史']) + '\n\n'
        texts = texts + get_json_value(record, ['入院记录', '病史小结', '查体']) + '\n\n'
        texts = texts + get_json_value(record, ['首次病程', '病例特点', '患者']) + '\n'
        texts = texts + get_json_value(record, ['首次病程', '病例特点', '现病史']) + '\n'
        texts = texts + get_json_value(record, ['首次病程', '病例特点', '既往史']) + '\n'
        texts = texts + get_json_value(record, ['首次病程', '病例特点', '查体'])

        entity_dict = {}

        print(texts)
        print(len(texts))
        anns = []
        ann_id = 0
        for ie_item in ie(texts):
            print(ie_item)
            """
            [{'症状': [{'end': 13,
                  'probability': 0.9998205973689949,
                  'relations': {'性质': [{'end': 18,
                                        'probability': 0.7157156076722941,
                                        'start': 16,
                                        'text': '加重'}]},
                  'start': 11,
                  'text': '呕吐'},
                 {'end': 11,
                  'probability': 0.9998343057367265,
                  'relations': {'性质': [{'end': 18,
                                        'probability': 0.6993950986271891,
                                        'start': 16,
                                        'text': '加重'}]},
                  'start': 9,
                  'text': '恶心'}]}]
            """
            if '症状' in ie_item:
                for item in ie_item['症状']:
                    entity_id = '%d_%d' % (tid, ann_id)
                    anns.append(transformer.assemble_ner_entity_ann(entity_id, item['start'], item['end'], item['text'], '症状', score=item['probability']))
                    ann_id = ann_id + 1

                    if 'relations' in item:
                        for rel in ['性质', '部位', '否定词']:
                            rel_distances = []
                            if rel in item['relations']:
                                for r_item in item['relations'][rel]:
                                    entity_key = '%s_%s_%s' % (r_item['start'], r_item['end'], r_item['text'])
                                    if entity_key not in entity_dict:
                                        r_entity_id = '%d_%d' % (tid, ann_id)
                                        ann_id = ann_id + 1
                                        anns.append(transformer.assemble_ner_entity_ann(r_entity_id, r_item['start'], r_item['end'], r_item['text'], rel, score=r_item['probability']))
                                        entity_dict[entity_key] = r_entity_id
                                    else:
                                        r_entity_id = entity_dict[entity_key]

                                    # 计算同类词的距离，选择最小
                                    rel_distances.append((r_entity_id, abs(r_item['start'] - item['start']), min(r_item['start'], item['start']), max(r_item['start'], item['start'])))

                                rel_distances = sorted(rel_distances, key=lambda x: x[1])
                                if texts[rel_distances[0][2]:rel_distances[0][3]].find('\n') == -1:
                                    anns.append(transformer.assembel_relation_entity_ann(rel_distances[0][0], entity_id, rel))
        results.append(transformer.assemble_anns(texts, anns))
        
    transformer.write_json_file(results, output)

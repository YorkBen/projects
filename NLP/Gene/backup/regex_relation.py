import sys
import argparse
import re

sys.path.append('../Lib/LabelStudio')

from LabelStudioTransformer import Transformer
from TrainDataGenerator import TrainDataGenerator

def gen_data(input):
    text_entitypair_relations = []
    ls_transfomer = Transformer()
    gen = TrainDataGenerator()
    for item in ls_transfomer.load_json_file(input):
        for item1 in gen.split_item_by_sep(item, separator='\n', append_sep=False):
            for item2 in gen.split_item_by_sep(item1, separator='. ', append_sep=True):
                entities = ls_transfomer.get_entities(item2)
                text = item2["data"]["text"].strip()
                if len(entities) < 2:
                    continue

                relations = ls_transfomer.get_relations(item2)
                if len(relations) == 0:
                    for e1 in entities:
                        for e2 in entities:
                            if e1[0] != e2[0]:
                                text_entitypair_relations.append((text, e1, e2, 'null'))
                else:
                    for from_id, to_id, label in relations:
                        r_entities = ls_transfomer.search_entities_in_list(entities, [from_id, to_id])
                        text_entitypair_relations.append((text, r_entities[0], r_entities[1], label))

    return text_entitypair_relations

def get_texts_between(text, entity_from, entity_to):
    start = min(entity_from[2], entity_to[2])
    end = max(entity_from[1], entity_to[1])
    search_text = text[start:end]
    return search_text

def check_abbrev(search_text, entity_from, entity_to):
    if entity_from[-2] == entity_to[-2]:
        return False

    search_text = search_text.strip()
    if len(search_text) > 0:
        if re.match('[(\-\[]{%s}' % len(search_text), search_text):
            return True
        else:
            return False
    else:
        if entity_to[-2][0] == '(':
            return True
        else:
            return False


def check_dependence(search_text):
    if re.match('-?dependent', search_text):
        return True
    else:
        return False


def check_responsive(search_text):
    if re.match('-?responsive', search_text):
        return True
    else:
        return False

def check_function(entity_from, entity_to):
    if entity_from[-1] == 'Gene' and entity_to[-1] == 'GeneFunction' \
        or entity_from[-1] == 'GeneFunction' and entity_to[-1] == 'Gene':
        return True
    else:
        return False

def check_multifunction(entity_from, entity_to):
    if entity_from[-1] == 'Gene' and entity_to[-1] == 'Gene multiFunction' \
        or entity_from[-1] == 'Gene multiFunction' and entity_to[-1] == 'Gene':
        return True
    else:
        return False

def check_target(search_text):
    if re.search('target', search_text):
        return True
    else:
        return False

def check_pathway(search_text):
    if re.search('in the', search_text):
        return True
    else:
        return False


def check_transcriptional_coactivation(search_text):
    if re.search('transcriptional coactivator', search_text):
        return True
    else:
        return False


def check_positive_negative(text):
    pos_ct = 0
    for match in re.findall('(upregula)|(up-regula)|(increase)|(induction)|(inducible)|(promot)|(elevate)|(overexpress)|(high level)|(high expression)', text):
        if match is not None:
            pos_ct = pos_ct + 1

    neg_ct = 0
    for match in re.findall('(downregula)|(down-regula)|(decrease)|(mutation)|(inactiv)|(deplet)|(loss of)|(low)|(repress)|(blockade)|(reduce)|(siRNA)|(suppress)|(inhibit)|(negtiave correlated)|(inverse correlation)|(inversely related to)', text):
        if match is not None:
            neg_ct = neg_ct + 1

    relate_ct = 0
    for match in re.findall('(significant correlation)|(correlated with)|(correlated significantly with)|(have an effect on)', text):
        if match is not None:
            relate_ct = relate_ct + 1

    if neg_ct > 0 and neg_ct % 2 != 0:
        return 'negative'
    elif pos_ct > 0:
        return 'positive'
    elif relate_ct > 0:
        return 'relatied'
    else:
        return 'null'


def check_relations(text, entity_from, entity_to):
    """
    检验关系类型
    """
    search_text = get_texts_between(text, entity_from, entity_to)
    label_from, label_to = entity_from[-1], entity_to[-1]
    # abbrev
    if label_from in ['Gene', 'MultiGene'] and label_to in ['Gene', 'MultiGene'] \
        and check_abbrev(search_text, entity_from, entity_to):
            return 'gene:abbrev'

    # dependence
    if label_from in ['Gene', 'MultiGene', 'signal pathway'] and label_to in ['Gene', 'MultiGene', 'signal pathway'] \
        and check_dependence(search_text):
            return 'gene:dependence'

    # responsive
    if label_from in ['Gene', 'MultiGene', 'signal pathway'] and label_to in ['Gene', 'MultiGene', 'signal pathway'] \
        and check_responsive(search_text):
            return 'gene:responsive'

    # target
    if label_from in ['Gene', 'MultiGene', 'signal pathway'] and label_to in ['Gene', 'MultiGene', 'signal pathway'] \
        and check_target(search_text):
            return 'gene:target'

    # transcriptional coactivation
    if label_from in ['Gene', 'MultiGene', 'signal pathway'] and label_to in ['Gene', 'MultiGene', 'signal pathway'] \
        and check_transcriptional_coactivation(search_text):
            return 'gene:transcriptional coactivation'

    # pathway
    if label_from in ['signal pathway'] and label_to in ['Gene', 'MultiGene', 'signal pathway'] \
        and check_pathway(search_text):
            return 'gene:pathway'

    # Gene function
    if check_function(entity_from, entity_to):
        return 'gene:Gene function'

    # Gene multifunction
    if check_multifunction(entity_from, entity_to):
        return 'gene:Gene multifunction'

    # positive, negative, relatied
    if check_positive_negative(text) == 'negative':
        return 'gene:negtive'
    elif check_positive_negative(text) == 'positive':
        return 'gene:positive'
    elif check_positive_negative(text) == 'relatied':
        return 'gene:relatied'
    else:
        return 'null'



    return 'null'



if __name__ =='__main__':
        # 参数
        parser = argparse.ArgumentParser(description='NER&RE TrainData generator parameters')
        parser.add_argument('-i', type=str, default='project.json', help='input file')
        args = parser.parse_args()

        input = args.i
        data = gen_data(input)
        check_label = 'gene:negtive'
        same_ct, not_match_ct, wrong_match_ct = 0, 0, 0
        for e in data:
            label = check_relations(e[0], e[1], e[2])

            # search_text = get_texts_between(e[0], e[1], e[2])
            # if e[-1] == check_label or label == check_label:
            #     print('')
            #     print(search_text)

            if e[-1] == check_label and label == check_label:
                same_ct = same_ct + 1
            elif e[-1] == check_label and label != check_label:
                print(e)
                not_match_ct = not_match_ct + 1
            elif e[-1] != check_label and label == check_label:
                wrong_match_ct = wrong_match_ct + 1
                print(e)

        print(same_ct, not_match_ct, wrong_match_ct)

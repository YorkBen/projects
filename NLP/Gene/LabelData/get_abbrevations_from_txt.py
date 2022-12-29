import sys
import re

sys.path.append('../../Lib/LabelStudio')

from LabelStudioTransformer import Transformer

entity_name_map = {
    "Gene": "Gene",
    "MultiGene": "Multiply_Gene",
    "Cancer": "Cancer",
    "MultiCancer": "Multiply_Cancer",
    "signal pathway": "Signal_Pathway",
    "GeneFunction": "Gene_Function",
    "Gene multiFunction": "Gene_Multi-Function"
}

relation_name_map = {
    "gene:positive": 'promotes',
    "gene:negtive": 'inhibits',
    "gene:relatied": 'is_related_to',
    "gene:promote dependence": "promotes_the_dependent_gene_of",
    "gene:inhibite dependence": "inhibits_the_dependent_gene_of",
    "gene:promote target": "promotes_the_target_gene_of",
    "gene:inhibite target": "inhibits_the_target_gene_of",
    "gene:dependence": "is_dependent_on",
    "gene:target": "'s_target_genes_is",
    "gene:transcriptional coactivation": "is_the_transcriptional_coactivation_of",
    "gene:promote pathway": "promotes_the_signaling_pathway_of",
    "gene:inhibite pathway": "inhibits_the_signaling_pathway_of",
    "gene:pathway": "'s_signaling_pathway_contains",
    "gene:responsive": "is_the_responsive_gene_of",
    "gene:positive relatied": "is_positively_related_to",
    "gene:negtive relatied": "is_negatively_related_to",
    "gene:upstream": "is_upstream_of",
    "gene:downstream": "is_downstream_of",
    "gene:Gene function": "act_as_the_function_of",
    "gene:Gene multifunction": "act_as_the_multi-function_of",

    "gene:abbrev": "is_the_abbreviation_of"
}

def replace_roma(s):
    s = s.replace('Ⅰ', 'I')
    s = s.replace('Ⅱ', 'II')
    s = s.replace('Ⅲ', 'III')
    s = s.replace('Ⅳ', 'IV')
    s = s.replace('Ⅴ', 'V')
    s = s.replace('Ⅵ', 'VI')
    s = s.replace('Ⅶ', 'VII')
    s = s.replace('Ⅷ', 'VIII')
    s = s.replace('Ⅸ', 'IX')
    s = s.replace('Ⅹ', 'X')
    s = s.replace('Ⅺ', 'XI')
    s = s.replace('Ⅻ', 'XII')

    return s

if __name__ == '__main__':
    t = Transformer()

    entity_type_list_dict = {}
    entity_dict = {}
    entity_ct_dict = {}
    relations = []
    for file in ['Data/project-1083.json']:
        for item in t.load_json_file(file):
            for e in t.get_entities(item):
                text = replace_roma(e[3]).replace('\t', ' ')
                if text[0] in ['(', '（'] and text[-1] in ['）', ')']:
                    text = text[1:-1]
                elif text[0] in ['(', '（'] and not re.search('[)）]', text):
                    text = text[1:]
                elif text[-1] in ['）', ')'] and not re.search('[(（]', text):
                    text = text[:-1]

                label = entity_name_map[e[4]]

                if label not in entity_type_list_dict:
                    entity_type_list_dict[label] = []
                entity_type_list_dict[label].append(text)
                entity_type_list_dict[label] = list(set(entity_type_list_dict[label]))

                entity_dict[e[0]] = (text, label)

                if text not in entity_ct_dict:
                    entity_ct_dict[text] = 0
                entity_ct_dict[text] += 1


            for r in t.get_relations(item):
                e0, e1 = entity_dict[r[0]], entity_dict[r[1]]
                e0_id = entity_type_list_dict[e0[1]].index(e0[0]) + 1
                e1_id = entity_type_list_dict[e1[1]].index(e1[0]) + 1
                label = relation_name_map[r[-1]]

                relations.append((e0_id, e0[0], e1_id, e1[0], label))

    # 写multi-Gene
    with open('Data/multi-gene.txt', 'w', encoding='utf-8') as f:
        for w in entity_type_list_dict['Multiply_Gene']:
            f.write('%s\n' % w)

    # 写multi-Cancer
    with open('Data/multi-cancer.txt', 'w', encoding='utf-8') as f:
        for w in entity_type_list_dict['Multiply_Cancer']:
            f.write('%s\n' % w)

    # 写Gene_Multi-Function
    with open('Data/multi-func.txt', 'w', encoding='utf-8') as f:
        for w in entity_type_list_dict['Gene_Multi-Function']:
            f.write('%s\n' % w)

    # 写abbrevation
    abbrevations = []
    for _, e1, _, e2, label in relations:
        if label == 'is_the_abbreviation_of':
            abbrevations.append('%s\t%s' % (e1, e2))
    abbrevations = sorted(list(set(sorted(abbrevations))))
    with open('Data/abbrevation.txt', 'w', encoding='utf-8') as f:
        for abbrev in abbrevations:
            f.write('%s\n' % abbrev)

    # 同义词处理
    synonyms = {}
    for line in abbrevations:
        e1, e2 = line.split('\t')
        if e1 not in synonyms and e2 not in synonyms:
            synonyms[e1] = [e1, e2]
            synonyms[e2] = synonyms[e1]
        elif e1 not in synonyms:
            synonyms[e2].append(e1)
            synonyms[e1] = synonyms[e2]
        elif e2 not in synonyms:
            synonyms[e1].append(e2)
            synonyms[e2] = synonyms[e1]

    synonyms_result = []
    for k, v in synonyms.items():
        v = sorted(v, key=lambda x: entity_ct_dict[x], reverse=True)
        synonyms_result.append((k, v[0]))
    synonyms_result = sorted(synonyms_result, key=lambda x: x[1])

    with open('Data/synonyms.txt', 'w', encoding='utf-8') as f:
        for k, v in synonyms_result:
            f.write('%s\t%s\n' % (k, v))

#

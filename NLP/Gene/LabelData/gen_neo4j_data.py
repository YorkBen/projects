import sys
import argparse
import os
from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "neo4j"))

sys.path.append('../../Lib/LabelStudio')

from LabelStudioTransformer import Transformer

"""将labelstudio导出的标记数据导入到neo4j数据库中"""

def create_constraint(tx):
    tx.run('CREATE CONSTRAINT cancerIdConstraint FOR (cancer:Cancer) REQUIRE cancer.id IS UNIQUE')
    tx.run('CREATE CONSTRAINT geneIdConstraint FOR (gene:Gene) REQUIRE gene.id IS UNIQUE')
    tx.run('CREATE CONSTRAINT genefuncIdConstraint FOR (genefunc:GeneFunction) REQUIRE genefunc.id IS UNIQUE')
    tx.run('CREATE CONSTRAINT multicancerIdConstraint FOR (multicancer:MultiplyCancer) REQUIRE multicancer.id IS UNIQUE')
    tx.run('CREATE CONSTRAINT multigeneIdConstraint FOR (multigene:MultiplyGene) REQUIRE multigene.id IS UNIQUE')
    tx.run('CREATE CONSTRAINT sigpathwIdConstraint FOR (sigpathw:SignalPathway) REQUIRE sigpathw.id IS UNIQUE')
    tx.run('CREATE CONSTRAINT multigenefuncIdConstraint FOR (multigenefunc:MultiplyGeneFunction) REQUIRE multigenefunc.id IS UNIQUE')

def create_node(tx, label, id, name):
    tx.run("CREATE (a:%s {id: toInteger($id), name: $name})" % label, id=id, name=name)

def create_relations(tx, text_from, label_from, text_to, label_to, rel_label):
    # tx.run("MATCH (f:%s), (t:%s)  WHERE f.name = '$text_from' and t.name = '$text_to' CREATE (a)-[r:IndexRel{type:$rel_label}]->(b)" % (label_from, label_to),
    tx.run("MATCH (f:%s), (t:%s)  WHERE f.id=toInteger(%s) and t.id=toInteger(%s) CREATE (f)-[r:%s]->(t)" % (label_from, label_to, text_from, text_to, rel_label))
    print("MATCH (f:%s), (t:%s)  WHERE f.id='%s' and t.id='%s' CREATE (f)-[r:%s]->(t)" % (label_from, label_to, text_from, text_to, rel_label))


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
    entity_type_list_dict = {}
    entity_dict = {}
    relations = []
    for item in t.load_json_file(input):
        for e in t.get_entities(item):
            text = e[3] if e[3][-1] != '）' else e[3][:-1]
            label = e[4]
            if label == 'Gene multiFunction':
                label = 'MultiplyGeneFunction'
            elif label == 'GeneFunction':
                label = 'GeneFunction'
            elif label == 'MultiCancer':
                label = 'MultiplyCancer'
            elif label == 'MultiGene':
                label = 'MultiplyGene'
            elif label == 'signal pathway':
                label = 'SignalPathway'

            if label not in entity_type_list_dict:
                entity_type_list_dict[label] = []

            if text not in entity_type_list_dict[label]:
                entity_type_list_dict[label].append(text)

            entity_dict[e[0]] = (text, label)


        for r in t.get_relations(item):
            e0, e1 = entity_dict[r[0]], entity_dict[r[1]]
            e0_id = entity_type_list_dict[e0[1]].index(e0[0]) + 1
            e1_id = entity_type_list_dict[e1[1]].index(e1[0]) + 1
            if r[-1] == "gene:positive":
                label = 'promote'
            elif r[-1] == "gene:negtive":
                label = 'inhibit'

            elif r[-1] == "gene:relatied":
                label = 'relatied'
                # label = 'is relatied to'
            elif r[-1] == "gene:positive relatied":
                label = 'positively_relatied'
                # label = 'is positively relatied to'
            elif r[-1] == "gene:negtive relatied":
                label = 'negtively_relatied'
                # label = 'is negtively relatied to'

            elif r[-1] == "gene:abbrev":
                label = 'abbreviation'
                # label = 'is abbreviation of'
            elif r[-1] == "gene:Gene function":
                label = 'Gene_Function'
                # label = 'is Gene function of'
            elif r[-1] == "gene:Gene multifunction":
                label = 'Gene_Function'
                # label = 'is Multiply Gene function of'

            elif r[-1] == "gene:pathway":
                label = 'pathway'
                # label = 'is pathway of'
            elif r[-1] == "gene:inhibite pathway":
                label = 'inhibite_pathway'
                # label = 'inhibite pathway of'

            elif r[-1] == "gene:promote pathway":
                label = 'promote_pathway'
                # label = 'promote pathway of'


            elif r[-1] == "gene:target":
                label = 'target'
                # label = 'is target of'
            elif r[-1] == "gene:inhibite target":
                label = 'inhibite_target'
                # label = 'inhibite target of'
            elif r[-1] == "gene:promote target":
                label = 'promote_target'
                # label = 'promote target of'

            elif r[-1] == "gene:dependence":
                label = 'dependence'
                # label = 'is dependent of'

            elif r[-1] == "gene:inhibite dependence":
                label = 'inhibite_dependence'
                # label = 'inhibite dependence of'

            elif r[-1] == "gene:promote dependence":
                label = 'promote_dependence'
                # label = 'promote dependence of'

            elif r[-1] == "gene:responsive":
                label = 'responsive'
                # label = 'is responsive to'
            elif r[-1] == "gene:transcriptional coactivation":
                label = 'transcriptional_coactivation'
                # label = 'is transcriptional coactivation of'


            elif r[-1] == "gene:upstream":
                label = 'upstream'
            elif r[-1] == "gene:downstream":
                label = 'downstream'

            else:
                label = r[-1]

            relations.append((e0_id, e0[1], e1_id, e1[1], label))

    relations = list(set(relations))


    # 写Neo4j
    with driver.session() as session:
        session.write_transaction(create_constraint)

        for label, v in entity_type_list_dict.items():
            for idx, n in enumerate(v):
                session.write_transaction(create_node, label, idx+1, n)

        for idx, r in enumerate(relations):
            session.write_transaction(create_relations, *r)

    driver.close()

    # 写文件
    # out_folder = r'Data/neo4j'
    # for k, v in entity_type_list_dict.items():
    #     with open(os.path.join(out_folder, '%s.csv' % k), 'w') as f:
    #         f.write('id\tname\n')
    #         for idx, n in enumerate(v):
    #             f.write('%s\t%s\n' % (idx+1, n))
    #
    # with open(os.path.join(out_folder, 'relations.csv'), 'w') as f:
    #     f.write('from_text\tfrom_label\tto_text\tto_label\trelation\n')
    #     for idx, r in enumerate(relations):
    #         f.write('%s\n' % ('\t'.join(r)))

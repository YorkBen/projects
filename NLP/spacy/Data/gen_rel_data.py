import sys
import argparse

sys.path.append('../../Lib/LabelStudio')

from TrainDataGenerator import TrainDataGenerator
gen = TrainDataGenerator()
# train_data = gen.process_gene_ner_forspacy(r'project-gene-200.json', 'max')
test_data = gen.process_gene_ner_forspacy(r'project-gene-250.json', 'default')
print(test_data[0])

import spacy
from spacy.tokens import DocBin
from spacy.symbols import ORTH
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, HYPHENS
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex, compile_suffix_regex

nlp = spacy.blank("en")
nlp.tokenizer.add_special_case("EGFRvIII", [{ORTH: "EGFR"}, {ORTH: "vIII"}])
nlp.tokenizer.add_special_case("wtEGFR", [{ORTH: "wt"}, {ORTH: "EGFR"}])
nlp.tokenizer.add_special_case("annexinV-FITC", [{ORTH: "annexin"}, {ORTH: "V-FITC"}])
nlp.tokenizer.add_special_case("124I-cG250", [{ORTH: "124I-c"}, {ORTH: "G250"}])
nlp.tokenizer.add_special_case("8)", [{ORTH: "8"}, {ORTH: ")"}])

# nlp.tokenizer.add_special_case("hsa_circ_PVT1", [{ORTH: "hsa_"}, {ORTH: "circ_"}, {ORTH: "PVT1"}])
# nlp.tokenizer.add_special_case("(ERalpha)-positive", [{ORTH: "("}, {ORTH: "ERalpha"}, {ORTH: ")"}, {ORTH: "-"}, {ORTH: "positive"}])


infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        r"(?<=.)[+\-\_()/](?=.)".format(a=ALPHA),
        # r"(?<=[{a}0-9])[+\-\_\)\(](?=[{a}0-9])".format(a=ALPHA),
    ]
)
infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

suffixes = nlp.Defaults.suffixes + [r"(?<=[{a}0-9])[-+$\)]".format(a=ALPHA)]
suffix_re = compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_re.search

# the DocBin will store the example documents
# for data, output in zip([train_data, test_data], ['train.spacy', 'test.spacy']):
for data, output in zip([test_data], ['train.spacy']):
    db = DocBin()
    for text, annotations in data:
        print(text)
        print(annotations)
        doc = nlp(text)\
        with doc.retokenize() as retokenizer:
            if doc[-1].text.endswith('.') and doc[-1].text != '.':
                retokenizer.split(doc[-1], [doc[-1].text[:-1], "."], heads=[doc[-1].head, (doc[-1], 0)])
        print([t.text for t in doc])
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label) # , alignment_mode='contract'
            ents.append(span)
        print(ents)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output)

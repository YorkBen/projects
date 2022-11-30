# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
from functools import partial

import paddle
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.utils.log import logger

from model import UIE
from utils import convert_example, reader, unify_prompt_name, create_data_loader

ner_labels = [
    "Multiply Gene",
    "Multiply Cancer",
    "Signal Pathway",
    "Gene Function",
    "Gene MultiFunction",
    "Cancer",
    "Gene"
]

relation_words = [
    "is positively related to",
    "is negatively related to",
    'is related to',
    "is dependent on",
    "is the responsive gene of",
    "promotes the dependent gene of",
    "inhibits the dependent gene of",
    "promotes the target gene of",
    "inhibits the target gene of",
    "is the transcriptional coactivation of",
    "promotes the signaling pathway of",
    "inhibits the signaling pathway of",
    "'s target genes is",
    "'s signaling pathway contains",
    "is upstream of",
    "is downstream of",
    "act as the function of",
    "act as the multi-function of",
    'promotes',
    'inhibits'
]

@paddle.no_grad()
def evaluate(model, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, pos_ids, start_ids, end_ids = batch
        start_prob, end_prob = model(input_ids, token_type_ids, att_mask,
                                     pos_ids)
        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')
        num_correct, num_infer, num_label, num_correct_record, num_record = metric.compute(
            start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label, num_correct_record, num_record)
    precision, recall, f1, record_acc = metric.accumulate()
    model.train()
    return precision, recall, f1, record_acc


def evaluate_subset(model, trans_fn, subset, type, key):
    test_data_loader = create_data_loader(subset,
                                          mode="test",
                                          batch_size=args.batch_size,
                                          trans_fn=trans_fn)

    metric = SpanEvaluator()
    precision, recall, f1, record_acc = evaluate(model, metric, test_data_loader)
    logger.info("-----------------------------")
    logger.info("%s：%s" % (type, key))
    logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f | Sample Num: %d" %
                (precision, recall, f1, len(subset)))


def do_eval():
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = UIE.from_pretrained(args.model_path)

    test_ds = load_dataset(reader,
                           data_path=args.test_path,
                           max_seq_len=args.max_seq_len,
                           lazy=False)

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_len=args.max_seq_len)

    ner_dict, relation_dict = {}, {}
    if args.debug:
        for data in test_ds:
            is_ner = False
            if len(data['result_list']) != 0:
                for ner_label in ner_labels:
                    if data['prompt'] == ner_label:
                        ner_dict.setdefault(ner_label, []).append(data)
                        is_ner = True
                        break

                if not is_ner:
                    for word in relation_words:
                        if data['prompt'].endswith(word):
                            relation_dict.setdefault(word, []).append(data)
                            break

        for key in ner_dict.keys():
            evaluate_subset(model, trans_fn, MapDataset(ner_dict[key]), "实体", key)

        for key in relation_dict.keys():
            evaluate_subset(model, trans_fn, MapDataset(relation_dict[key]), "关系", key)

    else:
        evaluate_subset(model, trans_fn, test_ds, "实体+关系", "all")



if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--debug", action='store_true', help="Precision, recall and F1 score are calculated for each class separately if this option is enabled.")

    args = parser.parse_args()
    # yapf: enable

    do_eval()

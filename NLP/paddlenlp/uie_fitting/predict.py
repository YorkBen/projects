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
from functools import partial

import paddle
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import get_span, get_bool_ids_greater_than

from model import UIE
from utils import convert_example, reader, unify_prompt_name, get_relation_type_dict, create_data_loader

golden_labels = ["急性阑尾炎", "急性胰腺炎", "肠梗阻", "异位妊娠", "急性胆管炎", "急性胆囊炎", "上尿路结石", "卵巢囊肿", "消化道穿孔"]


def text_reader(texts, prompt, max_seq_len=512):
    """
    read text line
    20221103 by pengxiang
    """
    for text in texts:
        json_line = {'content': text, 'prompt': prompt, 'result_list': [{"text": "", "start": 0, "end": 0}]}
        yield json_line


@paddle.no_grad()
def pred(model, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    results, labels = [], []
    model.eval()
    for batch in data_loader:
        input_ids, token_type_ids, att_mask, pos_ids, start_ids, end_ids = batch
        start_probs, end_probs = model(input_ids, token_type_ids, att_mask, pos_ids)

        start_ids = paddle.cast(start_ids, 'float32')
        end_ids = paddle.cast(end_ids, 'float32')

        pred_start_ids = get_bool_ids_greater_than(start_probs, return_prob=True)
        pred_end_ids = get_bool_ids_greater_than(end_probs, return_prob=True)
        lbl_start_ids = get_bool_ids_greater_than(start_ids)
        lbl_end_ids = get_bool_ids_greater_than(end_ids)

        for ps, pe, gs, ge in zip(pred_start_ids, pred_end_ids, lbl_start_ids, lbl_end_ids):
            pred_set = get_span(ps, pe, with_prob=True)
            label_set = get_span(gs, ge)

            results.append(list(pred_set))
            labels.append(list(label_set))

    return results, labels


def do_predict():
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = UIE.from_pretrained(args.model_path)

    test_ds = load_dataset(reader,
                           data_path=args.test_path,
                           max_seq_len=args.max_seq_len,
                           lazy=False)

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_len=args.max_seq_len)

    test_data_loader = create_data_loader(test_ds,
                                              mode="test",
                                              batch_size=args.batch_size,
                                              trans_fn=trans_fn)

    rs, ls = pred(model, test_data_loader)

    results, labels = [], []
    for r, l, s in zip(rs, ls, test_ds):
        results.append([tokenizer.decode(s[0][re[0][0]:re[1][0]+1]).replace(' ', '') + ':' + str((re[0][1] + re[1][1])/2) for re in r])
        labels.append([tokenizer.decode(s[0][re[0]:re[1]+1]).replace(' ', '') for re in l])

    with open('pred_result.txt', 'w', encoding="utf-8") as f:
        for line1, line2 in zip(results, labels):
            f.write(','.join(line1) + '\t' + ','.join(line2) + '\n')


def process_text(model, tokenizer, texts, prompt, max_seq_len=512, batch_size=8):
    """
    处理单行文本
    """
    test_ds = load_dataset(text_reader,
                           data_path=None,
                           max_seq_len=max_seq_len,
                           lazy=False,
                           texts=texts,
                           prompt=prompt)

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_len=max_seq_len)

    test_data_loader = create_data_loader(test_ds,
                                              mode="test",
                                              batch_size=batch_size,
                                              trans_fn=trans_fn)

    rs, ls = pred(model, test_data_loader)

    results, labels = [], []
    for r, l, s in zip(rs, ls, test_ds):
        results.append({tokenizer.decode(s[0][re[0][0]:re[1][0]+1]).replace(' ', ''):(re[0][1] + re[1][1])/2 for re in r})

    return results


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

    do_predict()

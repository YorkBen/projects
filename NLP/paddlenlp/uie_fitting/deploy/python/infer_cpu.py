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
import math
from pprint import pprint
import time

import paddle
from uie_predictor import UIEPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_path_prefix",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--position_prob",
        default=0.5,
        type=float,
        help="Probability threshold for start/end index probabiliry.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help=
        "The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="Batch size per CPU for inference.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    texts = [
        "0，0，1，0，2，2，0，0，2，0，0，0，0，0，0，0，0，1，1，2，0，1，2，0，2，0，0，0，0，1，31，2，1，2，0，2，2，2，2，2，0，2，0，1，2，2，2，2，2，2，2，2，2，2，0，0，0，0，0，2，2，2，0，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，0，0，0，0，0，2，2，2，0，2，2，2，2，2，2，2，2，2，2，2，1，2，2，0，2，0，2，2，0，0，0，0，0，0，0，0，0，0，0，0，0，0，1，2，2，2，2，2，2，0，0，0，0，2，0，2，2，2，3，2，2，2，2，2，1，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2",
        "0，0，1，0，2，0，0，0，2，0，0，0，0，0，0，1，0，0，2，2，0，2，2，0，2，0，0，0，0，1，33，2，0，2，0，2，2，2，2，2，0，2，0，0，2，2，2，2，2，2，2，2，2，2，0，0，0，0，0，2，2，2，0，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，0，0，0，0，0，2，2，2，0，2，2，2，2，2，2，2，2，2，2，2，1，2，2，2，2，0，2，2，0，0，0，0，0，0，0，0，1，0，0，0，0，0，1，1，2，0，2，2，2，0，0，0，0，0，0，2，2，2，2，2，2，2，2，2，2，2，2，1，2，2，2，2，2，1，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2，2"
    ]
    schema1 = '预测疾病[急性阑尾炎,异位妊娠,急性胆囊炎,急性胰腺炎,上尿路结石,卵巢囊肿,肠梗阻,消化道穿孔,急性胆管炎]'

    args.device = 'cpu'
    args.device_id = 1
    args.schema = schema1
    predictor = UIEPredictor(args)

    time1 = time.time()
    print("-----------------------------")
    outputs = predictor.predict(texts)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema1)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")
    time2 = time.time()
    print(time2 - time1)


if __name__ == "__main__":
    main()

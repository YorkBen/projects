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
        '患者半年余前进食油腻食物后出现中上腹疼痛不适，伴腹胀，无恶心、呕吐、腹泻、发热、黄疸等特殊不适，于我院急诊检查提示胆囊结石伴胆囊炎，予以消炎，解痉，止痛等对症处理后症状有所好转。后上述症状反复发作，均予以对症支持处理后有所缓解，1周前上述症状再发加重，伴尿黄，巩膜黄染，于我院急诊予以消炎解痉处理后症状缓解，现为求进一步治疗，来我院，门诊以“胆囊结石伴胆囊炎、胆总管扩张”收住入院。患者自起病来，神志清，精神可，大便正常，小便黄，体力体重无明显变化。',
        '患者4天无明显诱因出现腹痛，上腹为甚，伴有恶心、呕吐、呕吐物为胆汁样液体，伴停止排气排便，无腰背痛、发热、黄疸、头晕、胸痛、尿痛等不适，遂至我院就诊，考虑肠梗阻。行对症支持等保守治疗。后患者自行于阳逻当地医院就诊，行胃肠减压、抗炎补液等治疗，症状无明显缓解。现为求进一步诊治来院，急诊以“肠梗阻”收入院。起病以来，精神一般，食欲差，灌肠后解少量稀便，未自主排便，小便情况可，体重无明显变化。'
    ]

    schema1 = ['排尿改变[阴性,阳性,未知]']
    schema2 = ['厌食[未知,阳性,阴性]']

    args.device = 'cpu'
    args.device_id = 0
    args.schema = schema1
    predictor = UIEPredictor(args)

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

    # Reset schema
    predictor.set_schema(schema2)
    outputs = predictor.predict(texts)
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema2)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")


if __name__ == "__main__":
    main()

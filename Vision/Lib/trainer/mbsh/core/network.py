# -*- coding: utf-8 -*-
# filename: models.py
import os
import random

import numpy as np

from mbsh import logger
from mbsh.core.models import SmallModel, LabelResult
import cv2


def resize(rawimg, width=224, height=224):
    fx = float(width) / rawimg.shape[0]
    fy = float(height) / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((width, height, 3), dtype=np.uint8) * 0
    w = img.shape[1]
    h = img.shape[0]
    x = int((width - w) / 2)
    y = int((height - h) / 2)

    outimg[y:y + h, x:x + w] = img
    return outimg


class ModelResult(object):
    def __init__(self, success=True, label_results=None):
        if not label_results:
            label_results = []
        self.labelResults = label_results
        self.title = ''
        self.desc = ''
        self.fileName = None
        self.success = success
        self.score = 0
        self.status = 'running'
        self.img = None
        self.is_inside = False
        self.has_activate_parts = []
        self.case_id=None

    def add(self, other):
        self.labelResults.append(other)

    def get_label_result(self, label_name):
        for label_result in self.labelResults:
            if label_name == label_result.label_name:
                return label_result
        return None

    def update_label_result(self, label):
        old = self.get_label_result(label.label_name)
        if old is None:
            self.add(label)
        else:
            old.class_index = label.class_index
            old.class_name = label.class_name
            old.confidence = label.confidence

    def __str__(self):
        return 'title=' + self.title + ", desc=" + self.desc + "," + ",".join(
            [str(label) for label in self.labelResults])

    def model_2_json(self):
        return {"labelResults": self.labelResults, "title": self.title, "desc": self.desc,
                "fileName": self.fileName, "success": self.success, 'score': self.score, 'status': self.status,
                "is_inside": self.is_inside, "has_activate_parts": self.has_activate_parts,"case_id":self.case_id}


class WmsModel(object):
    def __init__(self, is_mock=False):
        self.width = 224
        self.height = 224
        self.models = []
        self.is_mock = is_mock

    def load_model_process(self):

        from mbsh import create_app
        app = create_app(os.getenv('FLASK_CONFIG') or 'default')
        app.app_context().push()
        self.models = SmallModel.query.filter().all()
        print('init models ,len=%s' % len(self.models))
        if self.is_mock:
            logger.info('model is running as mock mode')

        for _model in self.models:
            _model.load_model(_model.k_fold, self.is_mock)

        logger.info("init all models finished")

    def init(self):
        self.load_model_process()
        # pro = Process(target=self.load_model_process)
        # pro.start()

    def predict_img(self, img, img_params=None, include_models=None, case_type=0):
        result = ModelResult()
        if self.is_mock:
            result = self.predict_mock()
        else:
            try:
                for _model in self.models:
                    if include_models:
                        if _model.name not in include_models:
                            continue

                    label_result = _model.test_img(img, img_params=img_params, case_type=case_type)
                    if not label_result:
                        continue
                    result.add(label_result)
                    if _model.name == '内镜分类':

                        # 在体外或者不清晰时
                        if label_result.class_index in [0, 14]:
                            break
            except:
                logger.exception('predict_img error')

        for label_result in result.labelResults:
            if label_result.label_name == '内镜分类':
                valid_label = WmsModel._to_valid_label(label_result)
                result.add(valid_label)

                result.success = valid_label.class_index > 0

                clean_label = WmsModel._to_clean_label(label_result)
                result.add(clean_label)

                nbi_label = WmsModel._to_nbi_label(label_result)
                result.add(nbi_label)

                inside_label = WmsModel._to_inside_label(label_result)
                result.add(inside_label)

                surgery_label = WmsModel._to_surgery_label(label_result)
                result.add(surgery_label)

                mang_label = WmsModel._to_huimang_label(label_result)
                result.add(mang_label)

                lose_label = WmsModel._to_lose_label(label_result)
                result.add(lose_label)

                judge_label = WmsModel._to_judge_label(label_result)
                result.add(judge_label)

                mouth_label = WmsModel._to_mouth_label(label_result)
                result.add(mouth_label)

                stomach_label = self._to_stomach_label(label_result)
                result.add(stomach_label)

        return result

    @staticmethod
    def _to_valid_label(label_result):
        # class_index = 0，14 时分别为不合格和模糊不清，即无效图片，其余为有效图片（非肠即胃）
        class_index = 1

        if label_result.class_index in [0, 14] and label_result.confidence >= 90:
            class_index = 0
        if label_result.class_index in [5, 15, 16, 17, 18, 19, 20, 21, 22]:
            class_index = 2

        valid_label = LabelResult('是否合格', class_index=class_index, confidence=label_result.confidence)
        return valid_label

    @staticmethod
    def _to_huimang_label(label_result):
        # class_index = 16-回盲瓣 和 21-阑尾口时为回盲部
        class_index = 0
        confidence = label_result.confidence
        if label_result.class_index in [16, 21]:
            class_index = 1
            confidence = label_result.merge_confidence(classes=[16, 21])

        mang_label = LabelResult('回盲部识别', class_index=class_index, confidence=confidence)
        return mang_label

    @staticmethod
    def _to_clean_label(label_result):
        # class_index = 14 时为不清晰
        if label_result.class_index in [0, 14]:
            class_index = 0
            confidence = label_result.confidence
        else:
            class_index = 1
            confidence = int(100 * (1 - label_result.predictions[14]))

        clean_label = LabelResult('是否清晰', class_index=class_index, confidence=confidence)
        return clean_label

    @staticmethod
    def _to_inside_label(label_result):
        # class_index = 0 时为体外
        if label_result.class_index in [0]:
            class_index = 0
            confidence = label_result.confidence
        else:
            class_index = 1
            confidence = int(100 * (1 - label_result.predictions[0]))

        inside_label = LabelResult('是否体内', class_index=class_index, confidence=confidence)
        return inside_label

    @staticmethod
    def _to_surgery_label(label_result):
        # class_index = 23 时为手术中
        if label_result.class_index in [23]:
            class_index = 1
            confidence = label_result.confidence
        else:
            class_index = 0
            confidence = int(100 * (1 - label_result.predictions[23]))

        surgery_label = LabelResult('是否手术', class_index=class_index, confidence=confidence)
        return surgery_label

    def _to_stomach_label(self, label_result):
        predictions = label_result.predictions[:11]
        if label_result.class_index in [13, 14]:
            class_index = 0
            confidence = label_result.confidence
        else:
            class_index = int(np.argmax(predictions))
            confidence = int(np.max(predictions) * 100)

        class_name = self.get_model('胃部分类').desc_list[class_index]
        stomach_label = LabelResult('胃部分类', class_index=class_index, confidence=confidence, predictions=predictions,
                                    class_name=class_name)
        return stomach_label

    @staticmethod
    def _to_mouth_label(label_result):
        # class_index = 12 时口咽部
        if label_result.class_index in [12]:
            class_index = 1
            confidence = label_result.confidence
        else:
            class_index = 0
            confidence = int(100 * (1 - label_result.predictions[12]))

        mouth_label = LabelResult('口咽部识别', class_index=class_index, confidence=confidence)
        return mouth_label

    @staticmethod
    def _to_nbi_label(label_result):
        # class_index = 11 时为NBI放大
        if label_result.class_index in [11]:
            class_index = 1
            confidence = label_result.confidence
        else:
            class_index = 0
            confidence = int(100 * (1 - label_result.predictions[11]))

        nbi_label = LabelResult('NBI放大', class_index=class_index, confidence=confidence)
        return nbi_label

    @staticmethod
    def _to_judge_label(label_result):
        # 当预测结果不为肠部位时 为无法判断
        if label_result.class_index in [1, 2, 3, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22]:
            class_index = 0
            confidence = label_result.confidence
        else:
            class_index = 1
            confidence = label_result.confidence

        judge_label = LabelResult('无法判断', class_index=class_index, confidence=confidence)
        return judge_label

    @staticmethod
    def _to_lose_label(label_result):
        # class_index = 14 时为图像模糊，视野丢失
        if label_result.class_index in [14]:
            class_index = 0
            confidence = label_result.confidence
        else:
            class_index = 1
            confidence = int(100 * (1 - label_result.predictions[14]))

        lose_label = LabelResult('视野丢失', class_index=class_index, confidence=confidence)
        return lose_label

    def get_model(self, label_name):
        for model in self.models:
            if model.name == label_name:
                return model
        return None

    def predict_mock(self):
        result = ModelResult()
        result.success = True
        for _model in self.models:
            index = random.randint(0, len(_model.desc_list) - 1)
            label = LabelResult(_model.name, class_index=index, class_name=_model.desc_list[index],
                                confidence=random.randint(10, 100),
                                predictions=[random.random() for i in range(len(_model.desc_list))])
            result.add(label)
        # logger.debug("predict mock result %s " % (result))
        return result

from datetime import datetime, date
import os
import json
import re
import shutil
import cv2
import  numpy as np

from sqlalchemy import Column, Integer, String, PickleType, DateTime, LargeBinary, Boolean, Float
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import model_from_json


from mbsh import db, g_temp_dir, logger
from mbsh.core import cal_file_name
from mbsh.core.plot import Plot
from mbsh.core.yolo import YOLO


def list_page(self, pc):
    """
    自动分页
    :param pc {page: 0, pageSize: 50, filterParams: []}
    """
    page_size_ = int(pc['pageSize'])
    page_ = int(pc['page'])
    data_list = self.query.offset(page_size_ * page_).limit(page_size_).all()
    total = self.query.count()
    result = WebResult(True, total=total, data_list=data_list)
    return result


db.Model.list_page = list_page


def get_img_dir():
    created_at = datetime.now()
    time_tag = created_at.strftime('%Y-%m-%d')
    path_join = os.path.join(g_temp_dir, 'plots', time_tag)
    if not os.path.exists(path_join):
        os.makedirs(path_join, exist_ok=True)
    return path_join


def to_json(result):
    return json.dumps(result, cls=ModelEncoder)


class ModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "model_2_json"):
            return obj.model_2_json()
        data = None
        try:
            data = json.JSONEncoder.default(self, obj)
        except TypeError:
            logger.exception('JSONEncoder error')
        return data


class LabelResult(object):
    def __init__(self, label_name, class_name=None, class_index=0, confidence=0.0, tolerance=0.5, overlay=None
                 , img_path=None, predictions=None, overlay_box=None, has_activate=False, overlay_cam=None):
        if predictions is None:
            predictions = []
        self.label_name = label_name
        self.class_name = class_name if class_name else label_name
        self.confidence = confidence
        self.tolerance = tolerance
        self.class_index = class_index
        self.overlay = overlay
        self.overlay_box = overlay_box
        self.overlay_cam = overlay_cam
        self.predictions = predictions
        self.has_activate = has_activate
        self.img_path = img_path
        self.report_path = None
        # 触发的激活的label
        self.trigger_activate_label = None

    def merge_confidence(self, classes):
        """
        合并多个类别的置信度
        """
        conf = 0
        if self.predictions is None or len(self.predictions) < 1:
            return conf

        for class_index in classes:
            conf += self.predictions[class_index]
        conf = conf / np.sum(self.predictions)
        return int(conf * 100)

    def succeed(self):
        return self.confidence > self.tolerance

    def __str__(self):
        return " %s :%s, %s (%s)" % (self.label_name, self.class_index, self.class_name, self.confidence)

    def model_2_json(self):
        return {"labelName": self.label_name, "className": self.class_name, "confidence": self.confidence,
                "tolerance": self.tolerance, "classIndex": self.class_index, "overlay": self.overlay,
                "overlayBox": self.overlay_box, "has_activate": self.has_activate, "img_path": self.img_path,
                "report_path": getattr(self, "report_path", None), "overlay_cam": getattr(self, "overlay_cam", None),
                "trigger_activate_label": getattr(self, "trigger_activate_label", None)}


class TrainArgs(object):
    def __init__(self, shear_range=0, zoom_range=0,
                 shift_range=0, flip=False, rotation_range=0):
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.flip = flip

    def model_2_json(self):
        return {"shear_range": self.shear_range, "rotation_range": self.rotation_range, "zoom_range": self.zoom_range,
                "flip": self.flip, "shift_range": self.shift_range}


class SmallModel(db.Model):
    """
    单个模型定义，如病灶模型，部位模型等
    """
    __tablename__ = 't_small_model'
    name = Column(String(20), primary_key=True)
    desc_list = Column(PickleType)

    target_fold = Column(String(120))
    accuracy = Column(Integer, default=0)
    update_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_status = Column(String(20), default="success")
    number = Column(Integer, default=0)
    explain = Column(Boolean, default=False)
    k_fold = Column(Integer, default=1)
    train_args = Column(PickleType)
    tolerance = Column(Float, default=50.0)
    case_type = Column(Integer, default=0)

    def __init__(self, name, target_fold='', desc_list=None, number=1, tolerance=50, explain=False, k_fold=5,
                 train_args=TrainArgs(), case_type=0):
        if desc_list is None:
            desc_list = []
        self.models = []
        self.target_fold = target_fold
        self.name = name
        self.number = number
        self.tolerance = tolerance
        self.desc_list = desc_list

        self.accuracy = 0
        self.explain = explain
        self.k_fold = k_fold
        self.train_args = train_args
        self.explain_model = None
        self.case_type = case_type

    @staticmethod
    def merge_several_folds_mean(data):
        """
         通过求平均值合并模型结果
        """
        a = np.array(data[0])
        count = len(data)
        for i in range(1, count):
            a += np.array(data[i])
        a /= count
        return a.tolist()

    def predict(self, img, batch_size=32, verbose=0):
        all_result = []
        for k_num in range(0, len(self.models)):
            model = self.models[k_num]
            predictions = model.predict(img, batch_size=batch_size, verbose=verbose)

            all_result.append(predictions)
        return self.merge_several_folds_mean(all_result)

    def load_model(self, model_count=5, is_mock=False):
        if not is_mock:
            if self.name in ['息肉识别']:
                self.load_yolo3_model(model_count=model_count)
                # self.load_mask_model(model_count=model_count)
            else:
                # self.load_cnn_model(model_count=model_count)
                self.load_one_model(k_num=model_count)
        self.check_parents()

    def load_one_model(self, k_num=1):
        try:
            if not hasattr(self, 'models'):
                self.models = []
                self.explain_model = None

            file = self.target_fold + '/cache' + '/weights' + str(k_num) + '.hdf5'
            json_file = self.target_fold + '/cache' + '/model' + str(k_num) + '.json'
            if not os.path.exists(file):
                logger.info(" can not find file %s " % os.path.abspath(file))

            with open(json_file, 'r') as _file:
                model = model_from_json(_file.read())
                model.load_weights(file)
            logger.info("load model %s " % file)
            self.models.append(model)
            logger.info('load model %s finished' % self.name)
        except:
            logger.exception('loading model fail..')

    def load_cnn_model(self, model_count=5):
        try:
            if not hasattr(self, 'models'):
                self.models = []
                self.explain_model = None

            for k_num in range(1, model_count + 1):
                file = self.target_fold + '/cache' + '/weights' + str(k_num) + '.hdf5'
                json_file = self.target_fold + '/cache' + '/model' + str(k_num) + '.json'
                if not os.path.exists(file):
                    logger.info(" can not find file %s " % os.path.abspath(file))
                    continue
                with open(json_file, 'r') as _file:
                    model = model_from_json(_file.read())
                    model.load_weights(file)
                logger.info("load model %s " % file)
                self.models.append(model)
            logger.info('load model %s finished' % self.name)
        except:
            logger.exception('loading model fail..')

    def load_mask_model(self, model_count=1):
        from keras_retinanet import models
        try:
            if not hasattr(self, 'models'):
                self.models = []

            file = self.target_fold + '/cache' + '/model' + str(1) + '.h5'

            if not os.path.exists(file):
                logger.info(" can not find file %s " % os.path.abspath(file))
                return
            logger.info("  find file %s " % os.path.abspath(file))
            model = models.load_model(file, backbone_name='resnet50', convert=True)
            logger.info("load model %s " % file)
            self.models.append(model)
            # load cnn model for checking
            file = self.target_fold + '/cache' + '/weights' + str(2) + '.hdf5'
            json_file = self.target_fold + '/cache' + '/model' + str(2) + '.json'
            if not os.path.exists(file):
                logger.info(" can not find file %s " % os.path.abspath(file))
                return

            with open(json_file, 'r') as _file:
                model = model_from_json(_file.read())
                model.load_weights(file)
                self.models.append(model)
                logger.info("load model %s " % file)
            logger.info('load model %s finished' % self.name)
        except:

            logger.exception('loading model fail..')

    def load_yolo3_model(self, model_count=2):
        try:
            if not hasattr(self, 'models'):
                self.models = []

            file = self.target_fold + '/cache' + '/model' + str(1) + '.h5'
            if not os.path.exists(file):
                logger.info(" can not find file %s " % os.path.abspath(file))
                return
            logger.info("  find file %s " % os.path.abspath(file))
            model = YOLO(model_path=file)
            logger.info("load model %s " % file)
            self.models.append(model)
            # load cnn model for checking
            file = self.target_fold + '/cache' + '/weights' + str(2) + '.hdf5'
            json_file = self.target_fold + '/cache' + '/model' + str(2) + '.json'
            if not os.path.exists(file):
                logger.info(" can not find file %s " % os.path.abspath(file))
                return

            with open(json_file, 'r') as _file:
                model = model_from_json(_file.read())
                model.load_weights(file)
                self.models.append(model)
                logger.info("load model %s " % file)
            logger.info('load model %s finished' % self.name)
        except:
            logger.exception('loading model fail..')

    def predict_mask(self, img, img_params=None):
        from keras_retinanet.utils.image import preprocess_image, resize_image

        image = preprocess_image(img)
        size = min(image.shape[0], image.shape[1])
        image, scale = resize_image(image, min_side=360, max_side=800)

        # process image
        #         start = time.time()
        # image = cv2.resize(image, (1024, 1024))
        # print(image.shape)

        res = np.expand_dims(image, axis=0)
        boxes, scores, labels = self.models[0].predict(res)

        """
        标记多个息肉，输出score > 0.3的标记框
        """
        boxes /= scale
        boxeslist = []

        rate = 180 * 224 / size
        check_boxes = []
        check_scores = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            bbox = [float(x) * 224 / size for x in box]
            if score < 0.5:
                break
            if (bbox[2] - bbox[0]) > rate or (bbox[3] - bbox[1]) > rate:
                continue
            check_boxes.append(box)
            boxeslist.append(bbox)
            check_scores.append(score)

        check_list = self.check_mask(img, check_boxes, size, debug=True)
        out_list = []
        out_score = 0
        for i, check in enumerate(check_list):
            if check:
                out_list.append(boxeslist[i])
                out_score = max(out_score, check_scores[i])

        return out_list, float(out_score)

    def predict_yolo3(self, img, img_params=None):
        size = min(img.shape[0], img.shape[1])
        boxes, scores, labels = self.models[0].predict(img)
        logger.debug("boxes=%s,scores=%s,labels=%s" % (boxes, scores, labels))
        boxeslist = []
        rate = 180 * 224 / size
        check_boxes = []
        check_scores = []
        for box, score, label in zip(boxes, scores, labels):
            bbox = [float(x) * 224 / size for x in box]
            if score < 0.2:
                break
            if (bbox[2] - bbox[0]) > rate or (bbox[3] - bbox[1]) > rate:
                continue
            check_boxes.append(box)
            boxeslist.append(bbox)
            check_scores.append(score)
        check_list = self.check_mask(img, check_boxes, size, debug=True)
        out_list = []
        out_score = 0
        for i, cnn_score in enumerate(check_list):
            if cnn_score > 0.3:
                out_list.append(boxeslist[i])
                out_score = max(out_score, check_scores[i])
        return out_list, float(out_score)

    def check_mask(self, img, boxes_list, size=360, debug=False):
        from keras.applications.imagenet_utils import preprocess_input
        imgs = []
        check_list = [False] * len(boxes_list)
        for box in boxes_list:
            width = box[2] - box[0]
            height = box[3] - box[1]
            center = (int(box[0] + width / 2), int(box[1] + height / 2))
            m_size = min(max(width, height) * 1.3, 360)
            cut_half = int(m_size / 2)
            _img = img[max(0, center[1] - cut_half):center[1] + cut_half,
                   max(0, center[0] - cut_half):center[0] + cut_half]

            _img = cv2.resize(_img, (224, 224))
            imgs.append(_img)
        if len(imgs) == 0:
            return check_list
        data = np.asarray(imgs, "float64")
        data = preprocess_input(data)
        predictions = self.models[1].predict(data)

        for i in range(0, len(boxes_list)):

            check_list[i] = predictions[i]
            if debug:
                out_path = '%s/box/c_%.2f_%s' % (g_temp_dir, predictions[i], cal_file_name('2.jpg'))
                cv2.imwrite(out_path, imgs[i])

        return check_list

    def test_img(self, img, img_params=None, case_type=0):
        if not hasattr(self, "models") or len(self.models) == 0:
            return None

        if case_type == 0:
            if self.case_type == 1 or self.case_type == 2:
                return None

        if case_type == 1:
            if self.case_type == 2:
                return None

        if case_type == 2:
            if self.case_type == 1:
                return None

        overlay_box, cam_file, cam_img = None, None, None
        if self.name in ['息肉识别']:
            overlay_box, score = self.predict_yolo3(img)
            # overlay_box, score = self.predict_mask(img)
            predictions = [1 - score, score]
            item = score * 100
            index = 1 if len(overlay_box) > 0 else 0
        else:
            input_img_size=(int(self.models[0].input.shape[1]),int(self.models[0].input.shape[2]))
            img = cv2.resize(img, input_img_size)
            x = np.array(img.copy())
            input = np.expand_dims(x, axis=0)
            predictions = self.predict(input)[0]

            if len(predictions) == 1:
                v = predictions[0] * 100

                item, index = (v, 1) if v > self.tolerance else (100 - v, 0)
                if self.name in ['早癌识别', '肠癌识别'] and index == 1:
                    box, cam_file, cam_img = self.predict_and_explain(input, predictions[0], img_params)
                    overlay_box = [box]
            else:
                item = int(np.max(predictions) * 100)
                index = int(np.argmax(predictions))

        r = LabelResult(self.name, class_name=self.desc_list[index], class_index=index, confidence=item,
                        overlay=cam_file, overlay_cam=cam_img, overlay_box=overlay_box, predictions=predictions)
        return r

    def predict_and_explain(self, img, val, img_params=None):
        # logger.debug("explain mode")
        model = self.models[0]
        weights = model.layers[177].get_weights()[0]
        if self.explain_model is None:
            self.explain_model = Model(model.input, [model.layers[172].output, model.output])

        out, predictions = self.explain_model.predict(img)
        out = out[0]
        y = predictions[0]
        cam = (val - 0.5) * np.matmul(out, weights)
        cam -= cam.min()
        cam /= cam.max()
        cam -= 0.2
        cam /= 0.8

        bbox, cam_file_path, cam_img = None, None, None
        if not img_params:
            img_params = {}
        overlay_type = img_params.get('overlay_type', 'cam_box')
        if overlay_type == 'cam':
            cam_img, cam_file_path = Plot.plot_cam(img[0], cam, get_img_dir())
        elif overlay_type == 'only_cam':
            cam_img = cam
        else:
            bbox = Plot.cal_cam_bbox(cam)

        return bbox, cam_file_path, cam_img

    def export_classes(self, predictions, src_files, out_path, min_level=0, max_level=1):
        hard_number = 0

        os.makedirs(out_path, exist_ok=True)
        for i in range(len(predictions)):
            if len(predictions[i]) == 1:
                index = 0 if predictions[i][0] < 0.5 else 1
            else:
                index = int(np.argmax(predictions[i]))

            class_out_path = os.path.join(out_path, str(index) + '-' + self.desc_list[index])
            if not os.path.exists(class_out_path):
                os.makedirs(class_out_path)

            if min_level < predictions[i][0] < max_level:
                hard_number += 1
                shutil.copy2(src_files[i], class_out_path)

        logger.info('export to classes dir %s ,%s' % (hard_number, hard_number / len(predictions)))

    def check_parents(self):
        self.parent_dict = {}
        self.parent_count = -1
        self.parent_desc_list = []

        parent_dict = {}
        parent_count = 0

        i = -1
        for x in range(0, len(self.desc_list)):
            fields = self.desc_list[x].split('#')
            key = fields[0]

            if key not in self.parent_desc_list:
                parent_count += 1
                self.parent_desc_list.append(key)
                i += 1
            parent_dict[x] = i

        if len(self.parent_desc_list) != len(parent_dict):
            self.parent_dict = parent_dict
            self.parent_count = parent_count
            logger.info('find parents dict {}'.format(parent_dict))

    def to_parent_prediction(self, child_prediction, parent_dict, parent_count):

        parent_prediction = [0] * parent_count
        for i, v in enumerate(child_prediction):
            parent_prediction[parent_dict[i]] += v

        return parent_prediction

    def sibling_count(self, class_index):
        """
        兄弟姐妹个数,独生子女为1
        """
        parent_index = self.parent_dict[class_index]
        times = 0
        for val in self.parent_dict.values():
            if parent_index == val:
                times += 1
        return times

    def fetch_parent_result(self, label_result):
        predictions = self.to_parent_prediction(label_result.predictions, self.parent_dict, self.parent_count)
        predictions = predictions / np.sum(predictions)
        class_index = int(np.argmax(predictions))
        confidence = np.max(predictions)

        # 大类和小类不一致
        parent_class_index = self.parent_dict[label_result.class_index]
        if parent_class_index != class_index:
            # logger.warn('parent_class_index not equal class_index %s' % (predictions))
            confidence *= 0.5

        parent_class_name = self.parent_desc_list[class_index]

        # 约定去掉数字，为父类名称
        parent_model_name = re.sub('[^\D]', '', self.name)
        parent_label = LabelResult(parent_model_name, class_name=parent_class_name, class_index=class_index,
                                   predictions=predictions, confidence=int(confidence * 100))

        return parent_label

    def __repr__(self):
        return json.dumps(self.model_2_json())

    def model_2_json(self):
        return {"name": self.name, "accuracy": self.accuracy, "target_fold": self.target_fold,
                "tolerance": self.tolerance,
                "update_time": str(self.update_time), 'last_status': self.last_status, 'desc_list': self.desc_list,
                'index_list': [x for x in range(len(self.desc_list))],
                "train_args": self.train_args.model_2_json()}


class Job(db.Model):
    __tablename__ = 'train_jobs'
    id = Column(Integer(), primary_key=True, autoincrement=True)
    # running,success,failed
    status = Column(String(50), nullable=False, default='running')
    model_name = Column(String(50), nullable=False)
    accuracy = Column(Integer, default=0)
    images_count = Column(Integer, default=0)
    create_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    seconds = Column(Integer, default=-1)

    def __init__(self, model_name, images_count=0):
        self.model_name = model_name
        self.images_count = images_count

    def __repr__(self):
        return json.dumps(self.model_2_json())

    def model_2_json(self):
        return {"id": self.id, "model_name": self.model_name, "accuracy": self.accuracy,
                "images_count": self.images_count, "status": self.status,
                "create_time": str(self.create_time), "seconds": self.seconds}


class WebResult(object):
    def __init__(self, success=True, data_list=None, message=None, total=None):
        if data_list is None:
            data_list = []
        self.success = success
        self.list = data_list
        self.message = message

        self.total = len(data_list) if total is None else total

    def model_2_json(self):
        return {"success": self.success, "list": self.list, 'message': self.message, 'total': self.total}


class ImageData(db.Model):
    __tablename__ = 'image_data'
    fileName = Column(String(50), primary_key=True)
    labels = Column(PickleType)

    def __init__(self, fileName, labels):
        self.fileName = fileName
        self.labels = labels


class ImageFile(db.Model):
    __tablename__ = 'image_files'
    fileName = Column(String(50), primary_key=True)
    data = Column(LargeBinary)

    def __init__(self, fileName, data):
        self.fileName = fileName
        self.data = data


class Merchant(db.Model):
    '''
    商户信息
    '''
    __tablename__ = 'tbl_merchant'
    id = db.Column(db.Integer, primary_key=True)
    merchant_id = Column(String(50), unique=True)
    password = Column(String(128))
    status = Column(Integer, nullable=False, default='1')
    merchant_name = Column(String(250), nullable=False)
    contact_phone = Column(String(50))
    contact_user = Column(String(50))
    contact_email = Column(String(50))
    create_time = Column(DateTime, nullable=False, default=datetime.now())
    remark = Column(String(250), nullable=True)

    def model_2_json(self):
        return {'id': self.id, "merchant_id": self.merchant_id, "status": self.status,
                'merchant_name': self.merchant_name,
                'contact_phone': self.contact_phone, 'contact_user': self.contact_user, 'password': self.password,
                'create_time': self.create_time.timestamp(), 'contact_email': self.contact_email, 'remark': self.remark}


class CasePaper(db.Model):
    __tablename__ = 'case_paper'
    case_id = Column(String(50), primary_key=True)
    device_id = Column(String(50), nullable=False)
    case_type = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default='running')
    score = Column(Integer, default=0)
    active_parts = Column(PickleType(), default=0)
    lesions = Column(PickleType(), nullable=False)
    parts = Column(PickleType(), nullable=False)
    create_time = Column(DateTime, nullable=False, default=datetime.now())
    cost_time = Column(Integer, default=0)
    beat_num = Column(Float, default=0)
    patient_name = Column(String(50), nullable=True)
    study_id = Column(String(50), nullable=True)
    merchant_id = Column(String(50), nullable=True)
    # 远程状态
    remote_status = Column(Integer, default=0)
    # 肠镜进镜时间
    enter_time = Column(Integer, default=0)
    # 肠镜退镜时间
    exit_time = Column(Integer, default=0)
    patient_age = Column(Integer, default=-1, nullable=True)
    patient_phone = Column(String(50), nullable=True)
    # 性别
    patient_gender = Column(Integer, default=-1, nullable=True)
    remark = Column(String(250), nullable=True)
    # 客户端模式，0 为普通模式 1为对照组模式
    client_mode = Column(Integer, default=0, nullable=True)
    # 滑镜次数
    slide_count = Column(Integer, default=0, nullable=True)
    # 回镜成功次数
    back_count = Column(Integer, default=0, nullable=True)
    # 回镜均速
    average_speed = Column(Integer, default=0, nullable=True)

    project_id = Column(Integer, default=-1, nullable=True)
    study_case_type = Column(Integer, default=0, nullable=True)

    def __init__(self, case_id, device_id, case_type=1, lesions=None, parts=None):
        if not parts:
            parts = []
        if not lesions:
            lesions = []
        self.lesions = lesions
        self.parts = parts
        self.case_id = case_id
        self.device_id = device_id
        self.case_type = case_type
        self.score = 0
        self.active_parts = []
        self.create_time = datetime.now()
        self.cost_time = 0
        self.enter_time = 0
        self.exit_time = 0
        self.slide_count = 0
        self.back_count = 0
        self.status = "running"
        self.beat_num = 0.0
        self.patient_name = None
        self.study_id = None
        self.average_speed = 0

    @staticmethod
    def fetch_papers(device_id=None):
        today = date.today()
        if device_id:
            papers = CasePaper.query.filter(CasePaper.device_id == device_id, CasePaper.create_time >= today).order_by(
                CasePaper.create_time.asc())
        else:
            papers = CasePaper.query.filter(CasePaper.create_time >= today).order_by(
                CasePaper.create_time.desc())

        return list(papers)

    @staticmethod
    def fetch_top_papers():
        today = date.today()

        papers = CasePaper.query.filter(CasePaper.create_time >= today).order_by(
            CasePaper.score.desc()).limit(20)

        return list(papers)

    def update_report_params(self, report_params=None):
        """
        更新数据库的报告参数
        """
        if not report_params:
            return
        self.exit_time = report_params.get('exit_time', 0)
        self.enter_time = report_params.get('enter_time', 0)
        self.cost_time = self.exit_time + self.enter_time
        self.slide_count = report_params.get('slide_count', 0)
        self.back_count = report_params.get('back_count', 0)
        self.average_speed = report_params.get('average_speed', 0)

        db.session.merge(self)
        db.session.commit()

    def update_patient_info(self, data):
        """
        更新患者信息
        """
        patient_name = data.get('PatientName')
        if self.patient_name != patient_name:
            logger.info("update client_info  %s" % data)
            self.study_id = data.get('Number')
            self.patient_name = patient_name
            self.patient_age = data.get('PatientAge', -1)
            db.session.merge(self)
            db.session.commit()
        if "endoscopic_diagnosis" in data:
            logger.info("update case_info  %s" % self.case_id)
            case_info = CaseInfo.query.get(self.case_id)
            if case_info is None:
                case_info = CaseInfo()
                case_info.case_info_id = self.case_id
                case_info.device_id = self.device_id
            for key in data:
                if hasattr(case_info, key):
                    try:
                        setattr(case_info, key, data[key])
                    except:
                        logger.exception("update case info  attr err ,key=%s,name=%s" % (key, data[key]))
            db.session.merge(case_info)
            db.session.commit()

    def export_word(self):
        from docx import Document
        from docx.enum.table import WD_TABLE_ALIGNMENT

        # 打开文档
        document = Document()
        # 加入不同等级的标题
        # document.add_heading(u'内镜精灵评估结果', 0)
        # document.add_heading(u'开始时间：', 1)
        document.add_heading(u'检测到的部位：', 0)

        report_path = get_img_dir()
        out_path = os.path.join(report_path, self.case_id + u'_paper.docx')

        # 增加表格
        table_cols = 5
        table = document.add_table(rows=0, cols=table_cols)
        table.autofit = True
        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        paper = CasePaper.query.get(self.case_id)

        count_rows = len(paper.parts) // table_cols + 1

        for i in range(5):
            row_cells = table.add_row().cells
            for j in range(table_cols):
                p = row_cells[j].add_paragraph(text=u'a000', style=None)
                r = p.add_run()
                r.text = "haha1234"
                #     p1 = row_cells[j].add_paragraph()
                #     r1 = p1.add_run()
                #     index = i * table_cols + j
                #     if(index<len(paper.parts)):
                #         part = paper.parts[index]
                #         if type(part) == LabelResult and os.path.exists(part.img_path):
                #             with open(part.img_path, 'rb') as f:
                #                 data = io.BytesIO(f.read())
                #                 r.add_picture(data, width=Cm(4))
                #                 info = str(part.class_index + 1)+ part.class_name
                #                 r1.text = info

        # document.add_page_break()

        # document.add_heading(u'可疑病灶', 1)
        # p = document.add_paragraph()
        # r = p.add_run()
        # for lesion in paper.lesions:
        #     if type(lesion) == LabelResult and os.path.exists(lesion.img_path):
        #         with open(lesion.img_path, 'rb') as f:
        #             data = io.BytesIO(f.read())
        #
        #             r.add_picture(data, width=Inches(1.25))

        document.save(out_path)
        logger.info("export word to %s" % out_path)
        return out_path

    def model_2_json(self):
        return {"case_id": self.case_id, "lesions": self.lesions, "parts": self.parts, "status": self.status,
                'create_time': self.create_time.timestamp(), "cost_time": self.cost_time, "enter_time": self.enter_time,
                "exit_time": self.exit_time, "beat_num": self.beat_num,
                "score": self.score, "device_id": self.device_id, "case_type": self.case_type,
                'study_id': self.study_id, 'patient_name': self.patient_name, 'client_mode': self.client_mode,
                'slide_count': self.slide_count, 'back_count': self.back_count, 'average_speed': self.average_speed}


class CaseRank(db.Model):
    __tablename__ = 'case_rank'
    id = Column(String(50), primary_key=True)
    score = Column(Integer, default=0)
    count = Column(Integer, default=0)
    case_type = Column(Integer, default=1)

    def __init__(self, count, score, case_type):
        self.score = score
        self.count = count
        self.case_type = case_type
        self.id = self.make_id(case_type, score)

    @staticmethod
    def make_id(case_type, score):
        return "%s_%s" % (case_type, score)

    @staticmethod
    def add_rank(case_type, score, count=1):
        logger.debug("add a score to rank list ,score=%s" % score)
        rank_id = CaseRank.make_id(case_type, score)
        rank = CaseRank.query.get(rank_id)
        if rank is None:
            rank = CaseRank(score, 0, case_type)
        rank.count += count
        db.session.merge(rank)
        db.session.commit()

    @staticmethod
    def init_ranks():
        """
        初始化排行榜
        """
        case_types = [1, 2]
        for case_type in case_types:
            ranks = CaseRank.fetch_ranks(case_type)
            if len(ranks) < 100:
                logger.info("init ranks %s,%s ", case_type, ranks)
                for i in range(0, 101):
                    rank = CaseRank(0, i, case_type)
                    db.session.add(rank)
            db.session.commit()

    @staticmethod
    def fetch_ranks(case_type):
        ranks = list(CaseRank.query.filter(CaseRank.case_type == case_type).all())
        return ranks

    @staticmethod
    def fetch_beat_num(case_type, score):
        """
        获取本次分数对应击败比例
        """
        ranks = CaseRank.fetch_ranks(case_type)
        num = 0
        total = 0
        for rank in ranks:
            if rank.score < score:
                num += rank.count
            total += rank.count

        return 0


import hashlib
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask import current_app, url_for
from flask_login import UserMixin, AnonymousUserMixin



class Permission:
    FOLLOW = 1
    COMMENT = 2
    WRITE = 4
    MODERATE = 8
    ADMIN = 16


class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    default = db.Column(db.Boolean, default=False, index=True)
    permissions = db.Column(db.Integer)
    users = db.relationship('User', backref='role', lazy='dynamic')

    def __init__(self, **kwargs):
        super(Role, self).__init__(**kwargs)
        if self.permissions is None:
            self.permissions = 0

    @staticmethod
    def insert_roles():
        roles = {
            'User': [Permission.FOLLOW, Permission.COMMENT, Permission.WRITE],
            'Moderator': [Permission.FOLLOW, Permission.COMMENT,
                          Permission.WRITE, Permission.MODERATE],
            'Administrator': [Permission.FOLLOW, Permission.COMMENT,
                              Permission.WRITE, Permission.MODERATE,
                              Permission.ADMIN],
        }
        default_role = 'User'
        for r in roles:
            role = Role.query.filter_by(name=r).first()
            if role is None:
                role = Role(name=r)
            role.reset_permissions()
            for perm in roles[r]:
                role.add_permission(perm)
            role.default = (role.name == default_role)
            db.session.add(role)
        db.session.commit()

    def add_permission(self, perm):
        if not self.has_permission(perm):
            self.permissions += perm

    def remove_permission(self, perm):
        if self.has_permission(perm):
            self.permissions -= perm

    def reset_permissions(self):
        self.permissions = 0

    def has_permission(self, perm):
        return self.permissions & perm == perm

    def __repr__(self):
        return '<Role %r>' % self.name

    def model_2_json(self):
        return {'name': self.name, 'id': self.id}


class User(UserMixin, db.Model):
    __tablename__ = 'client_users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    password_hash = db.Column(db.String(128))
    confirmed = db.Column(db.Boolean, default=True)
    name = db.Column(db.String(64))
    location = db.Column(db.String(64))
    about_me = db.Column(db.Text())
    member_since = db.Column(db.DateTime(), default=datetime.utcnow)
    last_seen = db.Column(db.DateTime(), default=datetime.utcnow)
    avatar_hash = db.Column(db.String(32))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if self.role is None:
            if self.email == current_app.config['FLASKY_ADMIN']:
                self.role = Role.query.filter_by(name='Administrator').first()
            if self.role is None:
                self.role = Role.query.filter_by(default=True).first()
        if self.email is not None and self.avatar_hash is None:
            self.avatar_hash = self.gravatar_hash()

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_confirmation_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'confirm': self.id}).decode('utf-8')

    def confirm(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token.encode('utf-8'))
        except:
            return False
        if data.get('confirm') != self.id:
            return False
        self.confirmed = True
        db.session.add(self)
        return True

    def generate_reset_token(self, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'reset': self.id}).decode('utf-8')

    @staticmethod
    def reset_password(token, new_password):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token.encode('utf-8'))
        except:
            return False
        user = User.query.get(data.get('reset'))
        if user is None:
            return False
        user.password = new_password
        db.session.add(user)
        return True

    def generate_email_change_token(self, new_email, expiration=3600):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps(
            {'change_email': self.id, 'new_email': new_email}).decode('utf-8')

    def change_email(self, token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token.encode('utf-8'))
        except:
            return False
        if data.get('change_email') != self.id:
            return False
        new_email = data.get('new_email')
        if new_email is None:
            return False
        if self.query.filter_by(email=new_email).first() is not None:
            return False
        self.email = new_email
        self.avatar_hash = self.gravatar_hash()
        db.session.add(self)
        return True

    def can(self, perm):
        return self.role is not None and self.role.has_permission(perm)

    def is_administrator(self):
        return self.can(Permission.ADMIN)

    def ping(self):
        self.last_seen = datetime.utcnow()
        db.session.add(self)

    def gravatar_hash(self):
        return hashlib.md5(self.email.lower().encode('utf-8')).hexdigest()

    def gravatar(self, size=100, default='identicon', rating='g'):
        url = 'https://secure.gravatar.com/avatar'
        hash = self.avatar_hash or self.gravatar_hash()
        return '{url}/{hash}?s={size}&d={default}&r={rating}'.format(
            url=url, hash=hash, size=size, default=default, rating=rating)

    def model_2_json(self):
        json_user = {
            'id': self.id,
            'url': url_for('main.get_user', id=self.id),
            'username': self.username,
            'is_admin': self.is_administrator(),
            'member_since': self.member_since.timestamp(),
            'last_seen': self.last_seen.timestamp(),
            'role': self.role,
            'email': self.email

        }
        return json_user

    def generate_auth_token(self, expiration):
        s = Serializer(current_app.config['SECRET_KEY'],
                       expires_in=expiration)
        return s.dumps({'id': self.id}).decode('utf-8')

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        return User.query.get(data['id'])

    def __repr__(self):
        return '<User %r>' % self.username


class AnonymousUser(AnonymousUserMixin):
    def can(self, permissions):
        return False

    def is_administrator(self):
        return False



class Log(db.Model):
    __tablename__ = 't_log'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64))
    content = db.Column(db.String(128))
    ip = db.Column(db.String(64))
    create_at = db.Column(db.DateTime(), default=datetime.utcnow)

    @staticmethod
    def add_log(username, content, ip):
        log = Log()
        log.username = username
        log.content = content
        log.ip = ip
        db.session.add(log)
        db.session.commit()

    def model_2_json(self):
        return {'username': self.username, 'id': self.id, 'content': self.content, 'ip': self.ip,
                'create_at': self.create_at.timestamp()}


class CaseInfo(db.Model):
    __tablename__ = 'case_info'
    # 唯一编号
    case_info_id = Column(String(50), primary_key=True)
    device_id = Column(String(50), nullable=False)

    create_time = Column(DateTime, nullable=False, default=datetime.now())
    patient_name = Column(String(50), nullable=True)
    # 检查流水号
    study_id = Column(String(50), nullable=True)
    merchant_id = Column(String(50), nullable=True)

    patient_age = Column(Integer, default=-1, nullable=True)
    # 胃镜 1, 肠镜2
    case_info_type = Column(Integer, default=1, nullable=False)
    # 普通、无痛
    apply_type = Column(String(50), nullable=True)
    patient_phone = Column(String(50), nullable=True)
    # 家庭住址
    address = Column(String(256), nullable=True)
    # 报告医生
    doctor = Column(String(50), nullable=True)
    # 助手
    assistant = Column(String(50), nullable=True)
    # 建议
    suggest = Column(String(256), nullable=True)
    # 性别
    patient_gender = Column(Integer, default=-1, nullable=True)
    # 病人来源（门诊/住院）
    source = Column(String(50), nullable=True)
    # 送检科室（胃肠外科、胸外科）
    original_departmant = Column(String(50), nullable=True)
    # 主诉
    chief_complaint = Column(String(512), nullable=True)
    # 临床诊断
    clinical_diagnosis = Column(String(512), nullable=True)
    # 幽门螺杆菌
    hp_infection = Column(String(256), nullable=True)
    # 血妊娠检查
    pregnancy_test = Column(String(256), nullable=True)
    # 手术史
    surgery_history = Column(String(256), nullable=True)
    # 药物过敏史
    da_history = Column(String(256), nullable=True)
    # 禁食情况
    absolute_diet = Column(String(50), nullable=True)
    # 病理报告
    pathology_report = Column(String(512), nullable=True)
    # 内镜型号
    endoscope_model = Column(String(50), nullable=True)
    # 胃准备良好
    gastric_preparation = Column(String(50), nullable=True)
    # 是否有梗阻 0否，1有
    obstruction = Column(Integer, default=0, nullable=True)
    # 是否取活检 0否，1有
    biopsy = Column(Integer, default=0, nullable=True)
    # 内镜下诊断
    endoscopic_diagnosis = Column(String(512), nullable=True)
    # 不良事件
    adverse_event = Column(String(256), nullable=True)
    # 合并用药
    concomitant_medication = Column(String(512), nullable=True)

    remark = Column(String(250), nullable=True)


class ImageLabel:
    def __init__(self, file_name, model_name, class_index=0, confidence=0.0):
        self.model_name = model_name
        self.file_name = file_name
        self.class_index = class_index
        self.confidence = confidence
        self.remark_list = []
        self.remark_class_index = 0

    def model_2_json(self):
        return {
            'model_name': self.model_name, 'file_name': self.file_name, 'class_index': self.class_index,
            'confidence': self.confidence, 'remark_list': self.remark_list,
            'remark_class_index': self.remark_class_index
        }


class Device(db.Model):
    __tablename__ = 'tbl_device'
    # 设备号
    device_id = Column(String(30), primary_key=True)
    # 医院内部编号
    name = Column(String(250), nullable=True)
    ip = Column(String(250), nullable=True)
    experiment_mode = Column(Integer(), default=0)

    def __init__(self, device_id, name="", experiment_mode=0, ip=None):
        self.device_id = device_id
        self.name = name
        self.experiment_mode = experiment_mode
        self.ip = ip

    def model_2_json(self):
        return {"device_id": self.device_id, "name": self.name,
                "experiment_mode": self.experiment_mode, "ip": self.ip}

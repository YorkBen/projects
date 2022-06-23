# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import load_model, Model
# from tensorflow.python.keras.layers import Input, Lambda
# from tensorflow.python.keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda

from mbsh.core.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_loss
from mbsh.core.yolo3.utils import letterbox_image, letterbox_image_wh
from mbsh import logger

from mbsh.core.model_encrypt import decrypt_file

# from keras.engine.topology import load_weights_from_hdf5_group
# try:
    # from tensorflow.python.keras.engine.saving import load_weights_from_hdf5_group
    # from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
# except ImportError:
    # from tensorflow.python.keras._impl.keras.engine.topology import load_weights_from_hdf5_group
# modified by pengxiang 20220510


def create_yolo_model(input_shape, anchors, num_classes, weights_path=None, freeze_body=2):
    h, w = input_shape  # 尺寸
    image_input = Input(shape=(w, h, 3))  # 图片输入格式
    num_anchors = len(anchors)  # anchor数量

    # YOLO的三种尺度，每个尺度的anchor数，类别数+边框4个+置信度1
    tmp_dict = {0: 32, 1: 16, 2: 8}
    y_true = [Input(shape=(h // tmp_dict[i], w // tmp_dict[i],
                           num_anchors // 3, num_classes + 5)) for i in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)  # model
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if weights_path:  # 加载预训练模型
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)  # 加载参数，跳过错误
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False  # 将其他层的训练关闭
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    arg = {'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments=arg)(model_body.output + y_true)

    model = Model(inputs=[model_body.input] + y_true, outputs=model_loss)  # 模型，inputs和outputs
#   plot_model(model, to_file=os.path.join('model_data', 'model.png'), show_shapes=True, show_layer_names=True)

    return model


def yolo_get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def yolo_get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


class YOLO(object):
    _defaults = {
        "model_path": None,
        "anchors_path": None, # basedir+'/mbsh/core/yolo3/yolo_anchors.txt',
        "classes_path": None, # basedir+'/mbsh/core/yolo3/yolo_classes.txt',
        "score": 0.1,
        "iou": 0.5,
        "model_image_size": (352, 352),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        # base=os.path.split(self.model_path)[0]
        # self.classes_path=os.path.join(base,"yolo_classes.txt")
        # self.anchors_path=os.path.join(base,"yolo_anchors.txt")
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

        # self.sess = tensorflow.Session()
        # self.sess.run(tensorflow.global_variables_initializer())

    def _get_class(self):
        return yolo_get_class(self.classes_path)

    def _get_anchors(self):
        return yolo_get_anchors(self.anchors_path)

    def generate(self):

        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5') or model_path.endswith('.h5.e'), 'Keras model or weights must be a .h5 file.'
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            logger.info("load yolo model %s" % model_path)
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            if model_path.endswith('.h5'):
                self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
            # else:
            #     print('kkkkkkkkkkkkk')
            #     f = decrypt_file(model_path, file_handle=True)
            #     logger.info("load decrypt yolo model %s" % model_path)
            #     load_weights_from_hdf5_group(f, self.yolo_model.layers)  # , reshape=reshape
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, timing=False):
        if timing:
            start = timer()

        src_img_w = image.width
        src_img_h = image.height

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (src_img_w - (src_img_w % 32), src_img_h - (src_img_h % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # out_boxes, out_scores, out_classes = self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [src_img_h, src_img_w],
        #         K.learning_phase(): 0
        #     })
        # modified by pengxiang 20220610
        input_image_shape = tf.constant([image.size[1], image.size[0]])
        out_boxes, out_scores, out_classes = yolo_eval(self.yolo_model(image_data),
                                                    self.anchors, len(self.class_names), input_image_shape,
                                                    score_threshold=self.score, iou_threshold=self.iou)
        # ###################################

        b_count = len(out_boxes)
        print('Found {} boxes for {}'.format(b_count, 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * src_img_h + 0.5).astype('int32'))
        thickness = (src_img_w + src_img_h) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            #             draw.rectangle(
            #                 [tuple(text_origin), tuple(text_origin + label_size)],
            #                 fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 255), font=font)
            del draw

        if timing:
            end = timer()
            print(end - start)

        return image, b_count


    def detect_image_s(self, image, timing=False):
        if timing:
            start = timer()

        image_w = image.width
        image_h = image.height

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image,pad = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
            boxed_image = letterbox_image(image, new_image_size)
        # boxed_image = image.resize((352, 352))
        print(boxed_image.size,'dddddddddd')
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.



        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image_h, image_w],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print(out_boxes, pad, image_w, image_h)
        dx = image_w * pad[0] / self.model_image_size[0]
        dy = image_h * pad[1] / self.model_image_size[1]
        for item in out_boxes:
            item[0] += dx
            item[2] += dx
            item[1] += dy
            item[3] += dy
        print(out_boxes)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image_h + 0.5).astype('int32'))
        thickness = (image_w + image_h) // 300

        img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        result_box = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 去除小黑框
            y1, x1, y2, x2 = box
            nbox = [x1,y1,x2,y2]
            v_box = self.is_valid_box(img_cv2,nbox)
            if not v_box:
                continue

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top -= 10
            left -= 10
            bottom += 10
            right += 10
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            del draw
            result_box.append([left, top, right, bottom, score, predicted_class])
            # return left + i, top + i, right - i, bottom - i
            # return image[left + i, top + i, right - i, bottom - i]
            #
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])
            #
            # # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            #     draw.rectangle(
            #         [left + i, top + i, right - i, bottom - i],
            #         outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # del draw

        if timing:
            end = timer()
            print(end - start)

        return result_box

    def is_valid_box(self, img, box):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if len(img) == 0 or len(box) == 0:
            return False
        target = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
        if len(target) == 0 or len(target[0]) == 0:
            return False

        avg = float(np.max(target)) - float(np.min(target))
        mean_target = np.mean(target)
        valid = avg > 65 and mean_target > 50

        return valid

    def predict_file(self, file, target_path, draw_rect=True, one_box=False,
                     cut=False, expand=0, cut_pad=None, chk_box=True, timing=False, min_area=None):
        _,filename = os.path.split(file)
        image = Image.open(file)
        image_w = image.width
        image_h = image.height

        if timing:
            start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # out_boxes, out_scores, out_classes = self.sess.run(
        #     [self.boxes, self.scores, self.classes],
        #     feed_dict={
        #         self.yolo_model.input: image_data,
        #         self.input_image_shape: [image_h, image_w],
        #         K.learning_phase(): 0
        #     })
        # modified by pengxiang 20220610
        input_image_shape = tf.constant([image.size[1], image.size[0]])
        out_boxes, out_scores, out_classes = yolo_eval(self.yolo_model(image_data), self.anchors,
                                                        len(self.class_names), input_image_shape,
                                                        score_threshold=self.score, iou_threshold=self.iou)
        # ##########################################

        if timing:
            end = timer()
            print(end - start)

        # print('Found {} boxes for {}'.format(len(out_boxes), filename))

        out_boxes = list(out_boxes)
        out_scores = list(out_scores)
        out_classes = list(out_classes)
        # print(out_boxes, out_scores, out_classes)

        if draw_rect:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image_h + 0.5).astype('int32'))
            thickness = (image_w + image_h) // 300
            draw = ImageDraw.Draw(image)

        result_box = []
        cut_image = None
        # for i, c in list(enumerate(out_classes)):
        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            y1, x1, y2, x2 = box

            y1 = max(0, np.floor(y1 + 0.5).astype('int32'))
            x1 = max(0, np.floor(x1 + 0.5).astype('int32'))
            y2 = min(image.size[1], np.floor(y2 + 0.5).astype('int32'))
            x2 = min(image.size[0], np.floor(x2 + 0.5).astype('int32'))

            if chk_box:
                v_box = self.is_valid_box(image, [x1, y1, x2, y2])
                # print(x1,y1,x2,y2,v_box, 'valid box')
                if not v_box:
                    continue

            if min_area:
                if min_area >= 1:
                    if (x2-x1)*(y2-y1) < min_area:
                        continue
                else:
                    if (x2-x1)*(y2-y1)/image_w/image_h < min_area:
                        continue

            result_box.append([x1, y1, x2, y2, score, predicted_class])

            # if expand:
            if isinstance(expand, int):
                cut_x1 = x1 - expand
                cut_y1 = y1 - expand
                cut_x2 = x2 + expand
                cut_y2 = y2 + expand
            else:
                cut_x1 = x1 * (1-expand)
                cut_y1 = y1 * (1-expand)
                cut_x2 = x2 * (1+expand)
                cut_y2 = y2 * (1+expand)

            cut_x1 = max(0, np.floor(cut_x1 + 0.5).astype('int32'))
            cut_y1 = max(0, np.floor(cut_y1 + 0.5).astype('int32'))
            cut_x2 = min(image_w, np.floor(cut_x2 + 0.5).astype('int32'))
            cut_y2 = min(image_h, np.floor(cut_y2 + 0.5).astype('int32'))

            if cut:
                cut_image = np.array(image) # cv2.cvtColor(, cv2.COLOR_RGB2BGR)
                cut_image = cut_image[cut_y1:cut_y2, cut_x1:cut_x2]
                if cut_pad:
                    cut_image = self.pad_resize_image(Image.fromarray(cut_image), cut_x2-cut_x1, cut_y2-cut_y1, cut_pad)

                filename_base,filename_ext = os.path.splitext(filename)
                cut_filename = filename_base + '_cut_%02d' % i + filename_ext
                cut_image = Image.fromarray(cut_image)
                cut_image.save(os.path.join(target_path, cut_filename))

            if draw_rect:
                label = '{} {:.2f}'.format(predicted_class, score)

                label_size = draw.textsize(label, font)
                if y1 - label_size[1] >= 0:
                    text_origin = np.array([x1, y1 - label_size[1]])
                else:
                    text_origin = np.array([x1, y1 + 1])

                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)

                for j in range(thickness):
                    draw.rectangle([x1 + j, y1 + j, x2 - j, y2 - j], outline=self.colors[c])

            if one_box:
                break

        if draw_rect:
            del draw
            # 只有识别出目标才保存
            if len(result_box):
                image.save(os.path.join(target_path, filename))

        return result_box,image,cut_image

    def pad_resize_image(self, img, img_w, img_h, new_size):
        new_image = Image.new('RGB', (new_size, new_size), (0, 0, 0))

        if img_w <= new_size and img_h <= new_size:
            point_x = (new_size - img_w) // 2
            point_y = (new_size - img_h) // 2
            new_image.paste(img, (point_x, point_y))
            # new_image.paste(image, ((nw - w) // 2, (nh - h) // 2))
        else:
            scale = min(new_size/img_w, new_size/img_h)
            new_img_w = int(img_w * scale)
            new_img_h = int(img_h * scale)
            img = img.resize((new_img_w, new_img_h), Image.BICUBIC)

            point_x = (new_size - new_img_w) // 2
            point_y = (new_size - new_img_h) // 2
            new_image.paste(img, (point_x, point_y))

        return np.array(new_image)


    def predict(self, img):
        # img = cv2.resize(image, (352, 352))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [img.shape[0],img.shape[1]],
                K.learning_phase(): 0
            })
        bbox = [(box[1], box[0], box[3], box[2]) for box in out_boxes]
        return bbox, out_scores, out_classes

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

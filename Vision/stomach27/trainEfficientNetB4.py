import os
import sys

import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, model_from_json
# 添加路径，以能正常导入mbsh、trainer
sys.path.insert(0, r"Lib\trainer")

from mbsh import create_app
from mbsh.core.models import SmallModel
from mbsh.core.images import read_to_pd,TrainArgs
from mbsh.core.plot import Plot

from trainer import Trainer

# model_id = 100 ResNet 模型 loss: 0.2732 - acc: 0.9096 - val_loss: 0.4109 - val_acc: 0.8740

import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu70%的显存
# config.gpu_options.allow_growth = True      #程序按需申请内存
# sess = tf.Session(config = config)

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


# 样本数据根目录
# 根目录下res中按分类标签存放各类型图像
# cache目录存放训练的结果模型
#
root_path = r"\\192.168.0.154\数据\口咽部\大部位27分类"

# 是否使用平衡数据做训练
use_increase = False

# 训练采用的模型名称
model_name = 'efficientnet'

# 模型保存id
model_id = 28015


IMG_SIZE = 224
img_size = (IMG_SIZE, IMG_SIZE)

# img_size = (360,360)
# 训练迭代次数
epochs = 3


# 训练每批次样本数
batch_size = 16


# app创建
app = create_app(os.getenv('FLASK_CONFIG') or 'default')
app.app_context().push()


cache_path = root_path + '/cache'
res_path = root_path + '/res'

# res数据分割后的训练和验证集目录
res_train_path = root_path + '/res_train'
res_test_path = root_path + '/res_val'

# 数据平衡后的训练和验证集目录
train_path = root_path + '/train'
test_path = root_path + '/val'

# 图片预测错误存放目录
pred_err_path = root_path + '/pred_err'


# trainer创建

# 从数据库获取model信息，如果数据库还未建立相应的记录可查询一个已存在的，再自行修改sm的desc_list
# sm = SmallModel.query.get(sm_name)

# 生成 trainer
sm_name = '胃27模型'
sm_desc_list =  ['无法判断', '食管', '贲门', '胃窦#大弯', '胃窦#后壁', '胃窦#前壁', '胃窦#小弯', '十二指肠球部',
                 '十二指肠降部', '正镜胃体下部#大弯', '正镜胃体下部#后壁', '正镜胃体下部#前壁', '正镜胃体下部#小弯',
                 '正镜胃体中上部#大弯', '正镜胃体中上部#后壁', '正镜胃体中上部#前壁', '正镜胃体中上部#小弯','倒镜胃底#大弯',
                 '倒镜胃底#后壁', '倒镜胃底#前壁', '倒镜胃底#小弯','倒镜胃体中上部#后壁', '倒镜胃体中上部#前壁',
                 '倒镜胃体中上部#小弯','倒镜胃角#后壁', '倒镜胃角#前壁', '倒镜胃角#小弯', '胃#咽部']
sm = SmallModel(sm_name)


sm.desc_list = sm_desc_list
trainer = Trainer(sm)
trainer.img_size = img_size
trainer.target_fold = root_path

# 类型个数
types_num = len(trainer.desc_list)

desc_list = [str(x) + '-' + trainer.desc_list[x] for x in range(0,types_num)]
print(desc_list)


def img_rgb2_bgr(img):
    return img[: , : , : : -1]

# 图片生成器
# https://keras-cn.readthedocs.io/en/latest/preprocessing/image/#imagedatagenerator
train_datagen =  ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
        fill_mode='constant',
#         vertical_flip=True,
#         horizontal_flip=True,
        preprocessing_function = img_rgb2_bgr,
    )

test_datagen =  ImageDataGenerator(
        fill_mode='constant',
        preprocessing_function = img_rgb2_bgr,
    )

# 训练数据与测试数据
cls_mode = 'binary' if types_num == 2 else 'categorical'
train_generator = train_datagen.flow_from_directory(
        res_train_path,
        target_size = img_size,
        batch_size = batch_size,
        class_mode = cls_mode)
test_generator = test_datagen.flow_from_directory(
        res_test_path,
        target_size = img_size,
        batch_size = batch_size,
        class_mode = cls_mode)

train_samples = train_generator.samples
valid_samples = test_generator.samples


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# Create the base model Resnet50
base_model = EfficientNetB4(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')
# base_model = MobileNetV2(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')
# 冻结所有层
base_model.trainable = False
# base_model.trainable = True


from tensorflow.keras.optimizers import SGD
setting = (1, 'sigmoid', 'binary_crossentropy') if types_num == 2 else (types_num, 'softmax', 'categorical_crossentropy')
# Add a classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
prediction = Dense(setting[0], activation=setting[1], name='dense')(x)

model = Model(base_model.input, prediction)


base_learning_rate = 0.0001
sgd = SGD(lr=base_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])



def fit_gen(model, model_id, epochs=5):
    class_mode = 'binary' if types_num == 2 else 'categorical'
    print("train from imgs model_id=%s ,class_mode=%s" % (model_id, class_mode))

    # 创建cache目录
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    weight_path = cache_path + '/weights' + str(model_id) + '.hdf5'

    # EarlyStoppingy原型：
    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    # ModelCheckpoint原型：
    # ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    check_point = ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    # callbacks设置
    callbacks = [early_stop, check_point]


    history = model.fit(
        train_generator,
        epochs = epochs,
        steps_per_epoch = train_samples // batch_size,
        validation_data = test_generator,
        validation_steps = valid_samples // batch_size,
        callbacks = callbacks)
    return history


history = fit_gen(model, model_id, epochs=20)

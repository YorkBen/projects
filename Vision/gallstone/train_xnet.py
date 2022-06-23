import os
import sys
import time
import pandas as pd

# 添加路径，以能正常导入mbsh、unet
sys.path.insert(0, r'../Lib/trainer')

# os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# os.environ["SM_FRAMEWORK"] = "tf.keras"

# unet code from https://github.com/zhixuhao/unet
# from mbsh.core.unet.model import *
# from mbsh.core.unet.data import *
from mbsh.core.unet.data import trainGenerator

# code from https://github.com/MrGiovanni/UNetPlusPlus
from mbsh.core.unet_pp.segmentation_models import Unet, Xnet
from mbsh.core.xnet import mini_unet_model,xnet_predict_file,draw_outline_points,draw_outline_rect,sparse_ctrs
from mbsh.core.images import read_to_pd, save_img_file, read_img_file, cv2FindContours

import numpy as np
import cv2
from PIL import Image
from imageio import imread

from keras.optimizers import *
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

def build_model(model_path=None):
    # 建立模型
    if use_unet:
        model = Unet(input_shape=input_size, backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose',
                    classes=1 if num_class == 2 else num_class,
                    activation=activation_type)
    else:
        model = Xnet(input_shape=input_size, backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose',
                    classes=1 if num_class == 2 else num_class,
                    activation=activation_type)

    if model_path:
        model.load_weights(model_path)

    return model


# unet or unet++
use_unet = False

# 分割类型数量,背景除外
to_pred_num = 1

# 分割类型数量决定的变量
num_class = 1 + to_pred_num
flag_multi_class = True if num_class > 2 else False
activation_type = 'softmax' if flag_multi_class else 'sigmoid'
loss_type = 'categorical_crossentropy' if flag_multi_class else 'binary_crossentropy'

# 图片尺寸
img_w = 512
img_h = 512
input_shape = (img_h, img_w)
# input_size
input_size = (img_h, img_w, 3)


# 训练数据父目录
data_path = r'data\train_xnet'
# 原图子目录
images_folder = 'images'
# 掩码图子目录
labels_folder = 'labels'

# 验证集原图子目录
valid_images_folder = 'valid_images'
# 验证集掩码图子目录
valid_labels_folder = 'valid_labels'

# 测试集原图子目录
test_images_folder = 'test_images'
# 测试集掩码图子目录
test_labels_folder = 'test_labels'

# 模型保存目录
cache_path = data_path + r'\cache'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)
model_name = 'U_' if use_unet else 'X_' + 'ca_region' + "_" + time.strftime("%Y%m%d%H%M", time.localtime(time.time())) + '_{epoch:02d}-{val_accuracy:.3f}.hdf5'
abs_model_name = os.path.join(cache_path, model_name)


batch_size = 2
epochs = 100

# 对继续训练的设置模型路径
model_path = None # r'J:\郑碧清暂存\静脉曲张\胃底静脉曲张/cache\X_ca_region_21-03-10_17_14_04-0.843.hdf5'


# 从训练目录读取所有图像文件，得到表格，4列，分别为流水号、subject、classname、img
# 用to_csv可以保存为csv文件
train_df = read_to_pd(os.path.join(data_path, images_folder))
valid_df = read_to_pd(os.path.join(data_path, valid_images_folder))
train_samples, valid_samples = train_df['img'].count(), valid_df['img'].count()
print('Train imgs num: %s, valid imgs num: %s' % (train_samples, valid_samples))

# 训练数据
train_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='nearest')

trainGene = trainGenerator(batch_size, data_path, images_folder, labels_folder, train_gen_args,
                        image_color_mode="rgb", mask_color_mode="grayscale",
                        num_class = num_class, flag_multi_class = flag_multi_class,
                        target_size=(img_h,img_w))

# 验证数据
valid_gen_args = dict(fill_mode='nearest')

validGene = trainGenerator(batch_size, data_path, valid_images_folder, valid_labels_folder, valid_gen_args,
                        image_color_mode="rgb", mask_color_mode="grayscale",
                        num_class = num_class, flag_multi_class = flag_multi_class,
                        target_size=(img_h,img_w))

# 建立模型
model = build_model(model_path)
model.compile(optimizer = Adam(lr = 1e-4), loss = loss_type, metrics = ['accuracy'])
model_checkpoint = ModelCheckpoint(abs_model_name, monitor='val_accuracy', verbose=2, save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5)
callbacks = [early_stop, model_checkpoint]
history = model.fit_generator(trainGene, steps_per_epoch=train_samples // batch_size, epochs=epochs,
                    validation_data=validGene, validation_steps=valid_samples // batch_size,
                    callbacks=callbacks, workers=1)

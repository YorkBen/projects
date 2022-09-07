from openpyxl import Workbook
import numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, datasets
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD


base_dir = r'D:\项目资料\基因表达\20220804'
genes_file = os.path.join(base_dir, 'Diff_genes_name.csv')
patients_file = os.path.join(base_dir, '最终纳入的患者.csv')
m0_genes_file = os.path.join(base_dir, 'm0_all.csv')
m1_genes_file = os.path.join(base_dir, 'm1_all.csv')

hidden_size = 1024
gene_num = 537
BATCH_SIZE = 8000
epochs = 100

# cache path
cache_path = r'D:\projects\Vision\GenesPred\ckpts'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)

def get_mdl_name():
    save_path = cache_path
    mdl_save_name = 'mdl_' + time.strftime("%Y%m%d", time.localtime(time.time())) + '_{epoch:02d}-{val_loss:.4f}.hdf5'
    abs_model_name = os.path.join(save_path, mdl_save_name)

    return abs_model_name


def scheduler(epoch, lr):
    if epoch > 2:
        lr = lr * tf.math.exp(-0.051)
    return lr
scheduler(21, 0.01)

early_stop = EarlyStopping(monitor='val_loss', patience=3)
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
def get_callbacks(unfrozen_layers=None):
    model_checkpoint = ModelCheckpoint(get_mdl_name(), monitor='val_loss', verbose=2, save_best_only=True)
    callbacks = [early_stop, model_checkpoint, scheduler_callback]
    return callbacks

def build_model():
    #     scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)

    inputs = tf.keras.Input(shape=(2048,))

    # 模型定义
    #     x = scale_layer(inputs)
    x = inputs
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularize with dropout
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Regularize with dropout
    x = tf.keras.layers.Dense(gene_num)(x)

    #     outputs = tf.keras.layers.Softmax()(x)
    outputs = x
    model = tf.keras.Model(inputs, outputs)

    model.summary()

    return model



def load_gene_data():
    # 400多个基因
    gene_dict = {}
    with open(genes_file) as f:
        for line in f.readlines()[1:]:
            arr = line.strip().split(',')
            gene_dict[arr[1]] = arr[2]
    gene_id_arr = [gene_id for gene_id in gene_dict.keys()]
    # print(gene_dict)

    # 加载实验对象
    ensemble_m0, ensemble_m1 = [], []
    with open(patients_file, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            arr = line.strip().split(',')
            if arr[2] == 'M0':
                ensemble_m0.append(arr[0])
            else:
                ensemble_m1.append(arr[0])
    # print(ensemble_m0, ensemble_m1)

    # M0
    gene_data_m0 = {}
    with open(m0_genes_file) as f:
        for idx, line in enumerate(f.readlines()):
            arr = line.strip().split(',')
            if idx == 0:
                for name in arr[1:]:
                    patient_id = name[:12]
                    if patient_id in ensemble_m0:
                        gene_data_m0[(patient_id, arr.index(name))] = [-1 for i in range(len(gene_id_arr))]
            else:
                gene_id = arr[0].split('.')[0]
                if gene_id in gene_dict:
                    gene_id_idx = gene_id_arr.index(gene_id)
                    for patient_id, pidx in gene_data_m0.keys():
                        gene_data_m0[(patient_id, pidx)][gene_id_idx] = arr[pidx]

    # M1
    gene_data_m1 = {}
    with open(m1_genes_file) as f:
        for idx, line in enumerate(f.readlines()):
            arr = line.strip().split(',')
            if idx == 0:
                for name in arr[1:]:
                    patient_id = name[:12]
                    if patient_id in ensemble_m1:
                        gene_data_m1[(patient_id, arr.index(name))] = [-1 for i in range(len(gene_id_arr))]
            else:
                gene_id = arr[0].split('.')[0]
                if gene_id in gene_dict:
                    gene_id_idx = gene_id_arr.index(gene_id)
                    for patient_id, pidx in gene_data_m1.keys():
                        gene_data_m1[(patient_id, pidx)][gene_id_idx] = arr[pidx]


    gene_data = {'M0': {}, 'M1': {}}
    for patient_id, pidx in gene_data_m0.keys():
        gene_data['M0'][patient_id] = gene_data_m0[(patient_id, pidx)]
    for patient_id, pidx in gene_data_m1.keys():
        gene_data['M1'][patient_id] = gene_data_m1[(patient_id, pidx)]

    print('基因字典：%s, M0: %s, M1: %s' % (len(gene_dict), len(gene_data['M0']), len(gene_data['M1'])))

    return gene_dict, gene_data

def get_features(file_path):
    features = []
    with open(file_path) as f:
        for line in f.readlines():
            features.append([float(e) for e in line.split(' ')])
    return features

def load_data_from_path(gene_data, feature_path):
    features, labels = [], []
    for subfolder in ['M0', 'M1']:
        print('subfolder: %s' % subfolder)
        folder = os.path.join(feature_path, subfolder)
        for file_name in os.listdir(folder):
            print('file_name: %s' % file_name)
            patient_id = file_name[:12]
            if patient_id not in gene_data[subfolder]:
                continue

            file_path = os.path.join(folder, file_name)
            features_ = get_features(file_path)
            features.extend(features_)

            gene_arr = gene_data[subfolder][patient_id]
            labels.extend([gene_arr for i in range(len(features_))])
    print("特征维度：%s, 标签维度：%s" % (len(features), len(labels)))
    return (features, labels)


if __name__ == '__main__':
    feature_base_path = r'\\192.168.0.60\public\模型算法组\训练数据\zhangkuo\基础AI\DX-color-result0727特征'
    train_feature_path = os.path.join(feature_base_path, 'train')
    test_feature_path = os.path.join(feature_base_path, 'test')

    print('加载基因数据...')
    gene_dict, gene_data = load_gene_data()
    # 训练集数据
    print('处理文件夹：%s' % train_feature_path)
    train_data = load_data_from_path(gene_data, train_feature_path)

    # 验证集数据
    print('处理文件夹：%s' % test_feature_path)
    val_data = load_data_from_path(gene_data, test_feature_path)


    # 定义模型
    model = build_model()
    model.compile(optimizer = Adam(learning_rate = 0.0001),
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics = ['mse'])


    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).repeat(2).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(BATCH_SIZE)
    # history = model.fit(train_dataset, steps_per_epoch=len(train_features) // BATCH_SIZE, epochs=epochs,
    #                     validation_data=val_dataset, validation_steps=len(val_features) // BATCH_SIZE,
    #                     callbacks=get_callbacks(), workers=1)

    history = model.fit(train_dataset, steps_per_epoch=len(train_data) // BATCH_SIZE, epochs=epochs,
                        validation_data=val_dataset, validation_steps=len(val_data) // BATCH_SIZE,
                        callbacks=get_callbacks(), workers=1)

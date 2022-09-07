import numpy as np
import os
import time
from PIL import Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, datasets
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

img_gen = ImageDataGenerator(fill_mode='nearest' ,samplewise_center=True ,samplewise_std_normalization=True)

input_shape=(224, 224, 3)
base_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape = input_shape
        )

def build_model():
    #     scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)

    # 模型定义
    #     x = scale_layer(inputs)
    x = inputs
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dropout(0.5)(x)  # Regularize with dropout
#     x = tf.keras.layers.Dense(num_classes)(x)
    #     outputs = tf.keras.layers.Softmax()(x)
    outputs = x
    model = tf.keras.Model(inputs, outputs)

    model.summary()

    return model


if __name__ == '__main__':
    model = build_model()
    batch_size = 16

    dirs = [r'E:\DX-color-result0702\train\M0', r'E:\DX-color-result0702\train\M1', \
            r'E:\DX-color-result0702\test\M0', r'E:\DX-color-result0702\test\M1']

    # 排除目录
    exclude_folders = []
    for dir_ in dirs:
        for file in os.listdir(dir_):
            file_path = os.path.join(dir_, file)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                exclude_folders.append(file.replace('.txt', ''))

    # 遍历目录
    for dir_ in dirs:
        for folder in os.listdir(dir_):
            if folder in exclude_folders:
                continue

            folder_path = os.path.join(dir_, folder)
            if os.path.isdir(folder_path):
                print('processing folder: %s' % folder)
                with open(folder_path + '.txt', 'w') as f:
                    images = [os.path.join(folder_path, image) for image in os.listdir(folder_path) \
                              if image.endswith('.jpg') or image.endswith('.png')]
                    step = 8000
                    for i in range(0, len(images) + step, step):
                        if i == 0:
                            continue
                        print('step: %s' % i)
                        img_arr = []
                        for image_path in images[i-step:i]:
                            fp = open(image_path, 'rb')
                            img = Image.open(fp)
                            img_size = img.size
                            if img_size[0] == 224 and img_size[1] == 224:
                                img_arr.append(np.array(img))
                            else:
                                print('Warning: %s Size: (%s, %s)' % (image_path, img_size[0], img_size[1]))
                            fp.close()
                        img_arr = np.array(img_arr)
                        print(img_arr.shape)

                        val_gen = img_gen.flow(img_arr,
                                          batch_size = batch_size,
                                          shuffle = False)

                        # 训练
                        results = model.predict(val_gen, batch_size=batch_size, verbose=1)
                        print(results.shape)
                        for l in results:
                            l = [str(e) for e in l]
                            f.write('%s\n' % ' '.join(l))

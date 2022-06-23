import sys
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold

# from tensorflow.python.keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.python.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model, model_from_json

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import *

from tensorflow.keras.applications.densenet import DenseNet169

from tensorflow.keras import optimizers

# from tensorflow.python.keras.applications.xception import Xception

from mbsh.core.model_encrypt import load_enc_model, save_enc_model
from mbsh.core.models import ImageData
from mbsh.core.images import *
from mbsh import logger

# from keras.optimizers import Adam
# from mbsh.core.efficientnet.tfkeras import EfficientNetB0,EfficientNetB4,EfficientNetB7
from mbsh.core.efficientnet.tfkeras import EfficientNetB4

sys.path.insert(1, '../')
np.random.seed(2016)

__author__ = 'hill'


def create_link(paths, prefix):
    for img_path in paths:
        m_path = img_path.split('/')

        m_label = m_path[-2]
        m_name = m_path[-1]
        if not os.path.exists(prefix + '/' + m_label):
            os.makedirs(prefix + '/' + m_label)
        link_name = prefix + '/' + m_label + '/' + m_name
        abs_path = os.path.abspath(img_path)
        # print(abs_path)
        # print(link_name)
        os.symlink(abs_path, link_name)
        # shutil.copyfile(abs_path,link_name)


def preprocess_input(x):
    from keras.applications.vgg16 import preprocess_input
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    data = X[0]
    data = np.array(data, dtype=np.uint8)
    return data


# def cbam1(inputs):
#     inputs_channels = int(inputs.shape[-1])
#     x = GlobalAveragePooling2D()(inputs)
#     x = Dense(int(inputs_channels / 4))(x)
#     x = Activation("relu")(x)
#     x = Dense(int(inputs_channels))(x)
#     x = Activation("softmax")(x)
#     x = Reshape((1, 1, inputs_channels))(x)
#     x = multiply([inputs, x])
#     return x
#
#
def cbam_block1(inputs):
    inputs_channels = int(inputs.shape[-1])
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(int(inputs_channels / 4))(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(int(inputs_channels))(x)
    x = tf.keras.layers.Activation("softmax")(x)
    x = tf.keras.layers.Reshape((1, 1, inputs_channels))(x)
    x = tf.keras.layers.multiply([inputs, x])
    return x


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        # channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        # channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    # assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def channel_attention(input_feature, ratio=8):
    channel = int(input_feature.shape[-1])
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block2(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, )
    return cbam_feature


class Trainer(object):
    thread = None

    def __init__(self, sm):
        self.labelName = sm.name
        self.file_paths = []
        self.train_df = None
        self.target_fold = sm.target_fold
        assert (len(sm.desc_list) > 0)
        self.desc_list = sm.desc_list

        self.folders = [str(i) for i in range(0, len(self.desc_list))]
        self.smallModel = sm
        self.nfolds = self.smallModel.k_fold
        self.train_args = self.smallModel.train_args
        self.model_type = 'resnet'
        self.img_size = (224, 224)

    def copy_res(self):

        respath = self.target_fold + "/res"
        if os.path.exists(respath):
            shutil.rmtree(respath)

        print('export images to %s ...' % respath)
        ImageData.export_files(self.smallModel.name, respath)

    # split_test < 1 按比例，split_test >= 1 按数量
    def split_train_test_fold(self, split_test=0):
        res_dir = self.target_fold + '/res'
        train_dir = self.target_fold + '/res_train'
        test_dir = self.target_fold + '/res_test'

        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(test_dir, ignore_errors=True)

        _files = os.listdir(res_dir)
        files = sort_by_file_name(_files)
        for file in files:
            class_files = os.listdir(res_dir + '/' + file)

            if not os.path.exists(test_dir + '/' + file):
                os.makedirs(test_dir + '/' + file)
            if not os.path.exists(train_dir + '/' + file):
                os.makedirs(train_dir + '/' + file)

            random.shuffle(class_files)
            num = round(len(class_files) * split_test) if split_test < 1 else split_test
            assert (num < len(class_files))

            for i in range(num):
                shutil.copy2(res_dir + '/' + file + '/' + class_files[i], test_dir + '/' + file)

            for i in range(num, len(class_files)):
                shutil.copy2(res_dir + '/' + file + '/' + class_files[i], train_dir + '/' + file)

        print('split train and test fold finished, res_dir: %s, split_test: %s' % (res_dir, split_test))

    def increase_folders(self, count=20000, split_test=0):

        file_count = count // len(self.desc_list)
        train_dir = self.target_fold + '/res_train'
        test_dir = self.target_fold + '/res_test'

        train_out = self.target_fold + '/train'
        test_out = self.target_fold + '/test'
        shutil.rmtree(train_out, ignore_errors=True)
        shutil.rmtree(test_out, ignore_errors=True)
        increase_fold_imgs(train_dir, train_out, int(file_count * (1 - split_test)), self.train_args, self.img_size)
        if split_test > 0:
            increase_fold_imgs(test_dir, test_out, int(file_count * split_test), self.train_args, self.img_size)

        print('increase folders finished')

    def create_model(self, unfreeze_layers=None, attention=None):
        input_tensor = Input((self.img_size[0], self.img_size[1], 3))
        setting = (1, 'sigmoid', 'binary_crossentropy') if len(self.folders) == 2 else (
            len(self.folders), 'softmax', 'categorical_crossentropy')
        if self.model_type == 'vgg':
            print('create vgg model')
            base_model = VGG16(input_tensor=input_tensor, weights='imagenet')

            fc2 = base_model.get_layer('fc2').output

            prediction = Dense(setting[0], activation=setting[1], name='dense')(fc2)
            model = Model(base_model.input, prediction)
            for layer in model.layers[:-3]:
                layer.trainable = False

            sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])

        elif self.model_type == 'resnet':
            print('create model resnet..., unfreeze ', unfreeze_layers if unfreeze_layers else 'None')

            base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
            # base_model = Xception(weights='imagenet', include_top=False)

            for layers in base_model.layers[:]:
                layers.trainable = False
            if unfreeze_layers is not None:
                for layers in base_model.layers[-unfreeze_layers:]:
                    layers.trainable = True

            x = base_model.output
            if attention:
                print('add attention layer: ', attention)
                if attention == 1:
                    x = cbam_block1(x)
                else:
                    # if attention == 2:
                    residual = Conv2D(2048, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
                    residual = BatchNormalization(axis=3)(residual)
                    cbam = cbam_block2(x)
                    x = add([x, residual, cbam])

                    if attention == 3:
                        x = cbam_block1(x)

            # else:
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            prediction = Dense(setting[0], activation=setting[1], name='dense1')(x)
            model = Model(base_model.input, prediction)

            model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=optimizers.Adam(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=Adam(lr=0.001), loss=setting[2], metrics=['accuracy'])



            # model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss=setting[2], metrics=['accuracy'])
            # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            # model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])
        elif self.model_type == 'mobinetv2':

            print('create model mobinetv2...')
            base_model = MobileNetV2(input_tensor=Input((224, 224, 3)), weights='imagenet', include_top=False)

            for layers in base_model.layers[:]:
                layers.trainable = False
            if unfreeze_layers is not None:
                for layers in base_model.layers[-unfreeze_layers:]:
                    layers.trainable = True

            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            prediction = Dense(setting[0], activation=setting[1], name='mobileNet')(x)
            model = Model(base_model.input, prediction)
            sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])

        elif self.model_type == 'mobinet':
            print('create model mobinet...')
            base_model = MobileNet(input_tensor=Input((224, 224, 3)), weights='imagenet', include_top=False)

            for layers in base_model.layers[:]:
                layers.trainable = False
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            prediction = Dense(setting[0], activation=setting[1], name='mobileNet')(x)
            model = Model(base_model.input, prediction)
            sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        elif self.model_type == 'densenet':
            print('create model densenet..., unfreeze ', unfreeze_layers if unfreeze_layers else 'None')
            base_model = DenseNet169(input_tensor=input_tensor, weights='imagenet', include_top=False)

            for layers in base_model.layers[:]:
                layers.trainable = False
            if unfreeze_layers is not None:
                for layers in base_model.layers[-unfreeze_layers:]:
                    layers.trainable = True

            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            prediction = Dense(setting[0], activation=setting[1], name='dense')(x)
            model = Model(base_model.input, prediction)
            # model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=setting[2], metrics=['accuracy'])
            sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])
        elif self.model_type == 'inception':
            print('create model InceptionV3...')
            base_model = InceptionV3(weights='imagenet', include_top=False)
            # model = base_model
            for layers in base_model.layers[:]:
                layers.trainable = False
            if unfreeze_layers is not None:
                for layers in base_model.layers[-unfreeze_layers:]:
                    layers.trainable = True

            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(setting[0], activation=setting[1], name='dense_predictions')(x)
            model = Model(base_model.input, predictions)
            model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])

            # model = base_model
        elif self.model_type == 'xception':
            print('create model Xception..., unfreeze ', unfreeze_layers if unfreeze_layers else 'None')
            # 冻结原模型全连接层
            base_model = Xception(weights='imagenet', include_top=False)
            # base_model = Xception(weights=None, include_top=False)

            # 重新配置全连接层
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(128)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.2)(x)
            predictions = Dense(setting[0], activation=setting[1])(x)
            with tf.device('/cpu:0'):
                model = Model(inputs=base_model.input, outputs=predictions)
            # 重新训练迁移模型
            for layers in base_model.layers[:]:
                layers.trainable = False
            for layer in base_model.layers[-unfreeze_layers:]:
                layer.trainable = True

            model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss=setting[2], metrics=['accuracy'])
        elif self.model_type == 'efficient':
            print('create model efficient...')

            base_model = EfficientNetB4(input_tensor=input_tensor, weights='imagenet', include_top=False)
            for layers in base_model.layers[:]:
                layers.trainable = False
            if unfreeze_layers is not None:
                for layers in base_model.layers[-unfreeze_layers:]:
                    layers.trainable = True
            else:
                for layers in base_model.layers[-3:]:
                    layers.trainable = True

            x = GlobalAveragePooling2D()(base_model.output)
            x = Dropout(0.5)(x)
            x = Flatten()(x)

            prediction = Dense(setting[0], activation=setting[1], name='dense')(x)
            model = Model(base_model.input, prediction)

            # model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss=setting[2], metrics=['accuracy'])
            model.compile(optimizer=optimizers.Adam(0.0001), loss=setting[2], metrics=['accuracy'])

        else:
            print('ERROR ', self.model_type)

            # parallel_model = multi_gpu_model(model, gpus=1)




            # for layers in base_model.layers[:]:
            #     layers.trainable = False
            # for layer in base_model.layers[:-3]:
            #     layer.trainable = False
            # for layer in base_model.layers[-3:]:
            #     layer.trainable = True
            # x = GlobalAveragePooling2D()(base_model.output)
            # x = Dense(1024, activation='relu')(x)
            # x = Dropout(0.4)(x)
            #
            # predictions = Dense(setting[0], activation=setting[1], name='dense_predictions')(x)

            # model = Model(base_model.input, predictions)
            # sgd = SGD(lr=1e-4, momentum=0.9)
            # model.compile(optimizer=sgd, loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.MomentumOptimizer(0.0001, 0.9), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.RMSPropOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.AdagradOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.AdadeltaOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.MomentumOptimizer(0.0001, 0.9, use_nesterov=True), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.FtrlOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.compile(optimizer=tf.train.GradientDescentOptimizer(0.0001), loss=setting[2], metrics=['accuracy'])
            # model.summary()
            # print(model.optimizer._unconditional_dependency_names['optimizer'])


        return model

    def read_img(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) #cv2.imread(path)
        img = cv2.resize(img, self.img_size, cv2.INTER_LINEAR)
        return img

    def save_cache(self, data, file_path):
        dir_name = os.path.dirname(file_path)
        if (not os.path.exists(dir_name)):
            os.makedirs(dir_name)

        file = open(file_path, 'wb')
        pickle.dump(data, file)
        file.close()

    def load_cache(self, file_path):
        data = dict()
        if (os.path.isfile(file_path)):
            file = open(file_path, 'rb')
            data = pickle.load(file)
            file.close()
        else:
            print('file not exist: ' + file_path)
        return data

    def load_train_data(self):

        self.train_df = read_to_pd(self.target_fold + '/train')

        self.file_paths = self.train_df['img'].values
        return self.file_paths

    def create_k_folds(self):

        random_state = 51
        k_fold_num = 5 if self.nfolds == 1 else self.nfolds
        kf = KFold(n_splits=k_fold_num, shuffle=True, random_state=random_state)
        num_fold = 0
        k_fold_test_indecis = []
        print('start create K-Folds ...')
        out_put_dir = self.target_fold + '/k_folds_' + str(self.nfolds)

        if os.path.exists(out_put_dir):
            shutil.rmtree(out_put_dir)
        os.makedirs(out_put_dir)

        for file_train_index, file_test_index in kf.split(self.file_paths):
            num_fold += 1
            fold_base = out_put_dir + '/' + str(num_fold)
            print('create fold links ' + str(num_fold) + ' ...')
            k_fold_test_indecis.append(file_test_index)
            create_link(self.file_paths[file_train_index], fold_base + '/train')
            create_link(self.file_paths[file_test_index], fold_base + '/test')

            if num_fold >= self.nfolds:
                break

        # save_cache(k_fold_test_indecis,fold_base+"/k_fold_test_indecis.bin")

        print('create ' + str(k_fold_num) + ' folds finished.')

    def get_json_file(self, k_num):
        return self.target_fold + '/cache' + '/model' + str(k_num) + '.json'

    def load_model1(self, k_num):
        model = None
        # try:
        _file = open(self.get_json_file(k_num), 'r')
        model = model_from_json(_file.read())
        model.load_weights(self.get_weight_file(k_num))
        logger.info('load weights %s' % k_num)
        # except:
        #     logger.error("load model fail %s" % k_num)
        return model

    def load_model(self, k_num):
        model = None
        _file = open(self.get_json_file(k_num), 'r')
        model = model_from_json(_file.read())
        model.load_weights(self.get_weight_file(k_num))
        logger.info('load weights %s' % k_num)
        # try:
        #     _file = open(self.get_json_file(k_num), 'r')
        #     model = model_from_json(_file.read())
        #     model.load_weights(self.get_weight_file(k_num))
        #     logger.info('load weights %s' % k_num)
        # except:
        #     logger.error("load model fail %s" % k_num)
        return model

    def save_model(self, model, k_num, save_weight=True):
        model_json = model.to_json()
        with open(self.get_json_file(k_num), "w") as json_file:
            json_file.write(model_json)
        if save_weight:
            try:
                model.save_weights(self.get_weight_file(k_num), save_format='h5')
            except:
                model.save_weights(self.get_weight_file(k_num))

        logger.info("save model success ,num=%s" % k_num)

    def get_weight_file(self, k_num):
        return self.target_fold + '/cache' + '/weights' + str(k_num) + '.hdf5'

    def predict(self, img, batch_size=32, verbose=0):
        all_result = []
        for k_num in range(0, self.nfolds):
            model = self.load_model(k_num + 1)
            if not model:
                continue
            predictions = model.predict(img, batch_size=batch_size, verbose=verbose)
            # 由于显存有限，每次预测之后及时释放缓存，否则会出现OOM
            del model
            K.clear_session()
            all_result.append(predictions)
        return self.smallModel.merge_several_folds_mean(all_result)

    def fit_models(self, epoch=5, start_num=1, create_new=True):
        if not os.path.exists(self.target_fold + '/cache'):
            os.mkdir(self.target_fold + '/cache')

        for k in range(start_num, self.nfolds + 1):
            if create_new:
                model = self.create_model(k)

                self.save_model(model, k, False)
            else:
                model = self.load_model(k)

            self.fit(model, k, epoch)

            del model
            K.clear_session()

    def load_data_from_txt(self, txt_file):
        x_test = []
        y_test = []
        file_test = []

        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line_list = line.split(',')
            img_path = line_list[0]
            cls = int(line_list[1])
            try:
                img = self.read_img(img_path)
                # 对4通道的图片，在内存加载时会出错:X = np.asarray(X)
                if img.shape[2] > 3:
                    print("error file shape: {}, {}".format(img_path, img.shape))
                    img = img[:, :, 0:3]
                    print("after file shape: {}, {}".format(img_path, img.shape))

                file_test.append(img_path)
                y_test.append(cls)
                x_test.append(img)
            except:
                print('error img ' + img_path)

        print('load data from txt finished ,count=' + str(len(y_test)))
        return x_test, y_test, file_test

    def load_fold_data(self, fold, fold_classes=True):
        x_test = []
        y_test = []
        file_test = []
        _dirs = os.listdir(fold)
        dirs = sort_by_file_name(_dirs)
        print('find ' + str(len(dirs)) + ' from ' + fold)
        i = 0
        for _dir in dirs:

            files = os.listdir(fold + "/" + _dir)

            for file in files:

                img_path = fold + "/" + _dir + "/" + file
                if os.path.isfile(img_path):
                    try:
                        img = self.read_img(img_path)
                        # 对4通道的图片，在内存加载时会出错:X = np.asarray(X)
                        if len(img.shape) > 2:
                            if img.shape[2] > 3:
                                print("error file shape: {}, {}".format(file, img.shape))
                                img = img[:,:,0:3]
                                print("after file shape: {}, {}".format(file, img.shape))

                        file_test.append(img_path)
                        if fold_classes:
                            y_test.append(int(_dir))
                        else:
                            y_test.append(int(i))
                        x_test.append(img)
                    except:
                        print('error img ' + img_path)
            i += 1

        print('load data from fold finished ,count=' + str(len(y_test)))

        return x_test, y_test, file_test

    def to_parent_predictions(self, child_predictions, childs, parent_count):
        parent_predictions = []
        for child_prediction in child_predictions:
            parent_prediction = [0] * parent_count
            for i, v in enumerate(child_prediction):
                parent_prediction[childs[i]] += v
            parent_predictions.append(parent_prediction)
        return parent_predictions

    # childs parent_count 用法举例：5分类（0,1,2,3,4）要合并3、4分类出精度和矩阵结果，childs=[0,1,2,3,3]，parent_count=4
    def predict_data(self, folds, model=None, childs=None, parent_count=None, preprocess_input=None,
                     binary_threshold=0.5, fold_classes=True, txt_file=None):
        x_test = []
        y_test = []
        file_test = []
        start_time = datetime.datetime.now().microsecond
        for fold in folds:
            if txt_file:
                _x_test, _y_test, _file_test = self.load_data_from_txt(txt_file)
            else:
                _x_test, _y_test, _file_test = self.load_fold_data(fold, fold_classes)
            x_test += _x_test
            y_test += _y_test
            file_test += _file_test
        x_test = np.array(x_test)
        if preprocess_input:
            x_test = preprocess_input(x_test)
        if model is None:
            print('model is None')
            predictions = self.predict(x_test, verbose=1)
        else:
            predictions = model.predict(x_test, verbose=1)

        if childs is not None:
            print('show predictions as parent')
            predictions = self.to_parent_predictions(predictions, childs, parent_count)
            y_test = [childs[x] for x in y_test]

        error_files = []
        error_predict = []
        success_list = []
        if len(self.desc_list) == 2:
            y_pred = [1 if predication[0] > binary_threshold else 0 for predication in predictions]
        else:
            y_pred = [np.argmax(predication) for predication in predictions]
        y_true = y_test

        err = 0
        for i in range(0, len(y_pred)):
            if (y_pred[i] != y_true[i]):
                err += 1
                error_files.append(file_test[i])
                error_predict.append(y_pred[i])
            else:
                success_list.append(file_test[i])

        acc = int((1 - err * 1.0 / len(y_pred)) * 100)

        cast_time = datetime.datetime.now().microsecond - start_time
        print("cast{}ms, fold: {} ,accurracy:{}".format(cast_time // 1000, folds, acc))

        return acc, y_pred, y_true, file_test, predictions

    def export_predict(self, fold, out_fold, min_level=0, max_level=1.0, model=None,target_class_index=None):
        x_test, _y_test, _file_test = load_one_fold_data(fold, size=128)
        if model is None:
            predictions = self.predict(np.array(x_test), verbose=1)
        else:
            predictions = model.predict(np.array(x_test), verbose=1)

        hard_number = 0
        out_file_list = []
        out_dir_list = []
        for i in range(len(x_test)):
            prediction = predictions[i]
            v = np.max(prediction)
            class_index = np.argmax(prediction)
            if len(prediction) == 1:
                v, class_index = (v, 1) if v > 0.5 else (1 - v, 0)
                # logger.info("v=%s,class_index=%s" % (v, class_index))

            if target_class_index:
                if class_index not in target_class_index:
                    continue

            if min_level < v < max_level:
                hard_number += 1
                out_put_path = out_fold + '/' + str(class_index) + '-' + self.desc_list[class_index]

                out_dir_list.append(out_put_path)
                out_file_list.append(_file_test[i])

        copy_files_2(out_file_list, out_dir_list)

        logger.info("export predict data count=%s ,out dir %s" % (hard_number, out_fold))

    def create_classes_fold(self, out_fold):
        for i in range(0, len(self.desc_list)):
            out_put_path = out_fold + '/' + str(i) + '-' + self.desc_list[i]
            if not os.path.exists(out_put_path):
                os.makedirs(out_put_path)

    @staticmethod
    def trans_files(error_list, other_fold):
        move_files = 0
        for file in error_list:
            files = file.split('/')
            index = files[-2]
            dirs = other_fold + '/' + index + '/'
            if (not os.path.exists(dirs)):
                os.mkdir(dirs)

            # shutil.copyfile(file,dirs+files[-1])
            dest = dirs + files[-1]

            if os.path.exists(file) and not os.path.exists(dest):
                move_files += 1
                shutil.move(file, dest)

        print('copy {} files to {} finished'.format(move_files, other_fold))

    @staticmethod
    def extract_files(error_list, y_pred, y_true, other_fold, extract_all=False, remove=False):
        if os.path.exists(other_fold):
            shutil.rmtree(other_fold)
        move_files = 0
        for i in range(0, len(error_list)):
            if y_true[i] != y_pred[i] or extract_all:
                file = error_list[i]
                files = file.split('/')
                index = files[-2]
                dirs = other_fold + '/' + str(y_true[i]) + '/' + str(y_pred[i]) + '/'
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                # shutil.copyfile(file,dirs+files[-1])
                dest = dirs + files[-1]

                if os.path.exists(file):
                    move_files += 1
                    shutil.copyfile(file, dest)
                    if remove:
                        os.remove(file)

        print('extract {} files to {} finished'.format(move_files, other_fold))

    @staticmethod
    def export_files(file_list, y_pred, other_fold):
        if os.path.exists(other_fold):
            shutil.rmtree(other_fold)
        move_files = 0
        for i in range(0, len(file_list)):

            file = file_list[i]
            files = file.split('/')
            dirs = other_fold + '/' + str(y_pred[i]) + '/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)

            # shutil.copyfile(file,dirs+files[-1])
            dest = dirs + files[-1]

            if os.path.exists(file):
                move_files += 1
                shutil.copyfile(file, dest)

        print('export {} files to {} finished'.format(move_files, other_fold))

    def save_encrypt_model(self, k_num):
        json_file = self.get_json_file(k_num)
        weight_file = self.get_weight_file(k_num)
        save_enc_model(json_file, weight_file)

    def load_encrypt_model(self, k_num):
        json_file = self.get_json_file(k_num)
        weight_file = self.get_weight_file(k_num)
        model = load_enc_model(json_file, weight_file)
        return model

    def __str__(self):
        return self.labelName

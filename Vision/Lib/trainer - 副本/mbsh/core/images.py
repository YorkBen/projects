import base64
import datetime
import time
import pickle
import random
import re
import traceback
import shutil
import os
import math

import pandas as pd

from tensorflow.python.keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import requests
from PIL import Image

from mbsh import logger
import cv2
from mbsh.core.models import TrainArgs, LabelResult

__author__ = 'gpu'


def shift_corners(img):
    rows, cols, _ = img.shape
    center_ratio = random.randint(8, 10) * 0.1
    offset = int(rows * (1 - center_ratio))
    return img[offset:rows - offset, offset:cols - offset]


def transparent_overlay(back, overlay_file, outdir, width=400, height=400):
    overlay = cv2.imread(overlay_file, cv2.IMREAD_UNCHANGED)

    scale_w = random.randint(5, 8) * 0.1
    scale_h = scale_w * random.randint(9, 11) * 0.1

    overlay = cv2.resize(overlay, (int(width * scale_w), int(height * scale_h)))

    h, w, _ = overlay.shape  # Size of foreground
    src = cv2.imread(back)
    src = shift_corners(src)
    src = cv2.resize(src, (width, height))
    rows, cols, _ = src.shape  # Size of background Image
    pos = (random.randint(rows // 8, rows // 3), random.randint(cols // 8, cols // 3))
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    alpha_rate = random.randint(60, 100) * 0.01
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) * alpha_rate  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]

    out_path = outdir + '/' + cal_file_name('2222.png')
    cv2.imwrite(out_path, src)
    return out_path


def increase_positive(positive_dir, false_dir, out_dir, count=1000):
    positive_files = os.listdir(positive_dir)
    false_files = os.listdir(false_dir)
    logger.info("increase_positive_img ...")
    for i in range(0, count):
        false_path = random.choice(false_files)
        positive_path = random.choice(positive_files)
        try:
            transparent_overlay(false_dir + '/' + false_path, positive_dir + '/' + positive_path, out_dir)
        except:
            print(
                'error increase_positive_img ' + false_dir + '/' + false_path + ',' + positive_dir + '/' + positive_path)


def read_img_file(path):
    try:
        with open(path, 'rb') as img_file:
            bytes = img_file.read()
            nparr = np.fromstring(bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except:
        traceback.print_exc()
        logger.error("err file %s " % path)
    return None


def save_img_file(path, img):
    try:
        with open(path, 'wb') as img_file:
            data = cv2.imencode('.jpg', img)[1]
            img_file.write(data)
    except:
        traceback.print_exc()
        logger.error("err file %s " % path)
    return None


def custom_file_sort(x, y):
    x = x.split('-')[0]
    y = y.split('-')[0]
    return int(x) > int(y)


def join_processes(processes):
    for pro in processes:
        pro.start()
    for pro in processes:
        pro.join()


def increase_fold_imgs(src_dir, train_dir, count, train_args=TrainArgs(), size=(360, 360)):
    # print('进入方法')
    _folders = os.listdir(src_dir)
    folders = sort_by_file_name(_folders)

    max_count = 10
    total = len(folders)

    for_cunt = math.ceil(total / max_count)
    for i in range(0, for_cunt):
        # print('进入循环')
        processes = []

        start = i * max_count
        end = start + max_count
        if end > total:
            end = total

        for fold in folders[start:end]:
            # print('进入内循环')
            respath = src_dir + '/' + fold
            logger.info('increase_imgs %s ...' % respath)
            p1 = Process(target=increase_imgs, args=(respath, train_dir + '/' + fold, count, size, train_args))
            processes.append(p1)

        join_processes(processes)


def increase_imgs(respath, outpath, count=1000, target_size=(360, 360), train_args=TrainArgs()):
    # temppath = respath + '/temp'
    # print('进入内方法')
    # if not os.path.exists(temppath):
    #     try:
    #         os.symlink(os.path.abspath(respath), temppath)
    #     except:
    #         logger.warn('symlink error ' + temppath)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    train_datagen = ImageDataGenerator(
        shear_range=train_args.shear_range,
        zoom_range=train_args.zoom_range,
        width_shift_range=train_args.shift_range,
        height_shift_range=train_args.shift_range,
        horizontal_flip=train_args.flip,
        vertical_flip=train_args.flip,
        rotation_range=train_args.rotation_range,
        fill_mode='constant'
    )
    # print(respath)
    # respath = r'D:\Projects\bali_fenxing\res_train\0'
    gen = train_datagen.flow_from_directory(os.path.split(respath)[0], save_to_dir=outpath, target_size=target_size, batch_size=100,classes=[str(os.path.split(respath)[-1])])
    logger.info(respath)
    total = 0
    # return 
    while total < count:
        x, y = next(gen)
        # print(len(x))
        total += len(x)

    # del extra imgs
    for (dirpath, dirnames, filenames) in os.walk(outpath):
        k = 0
        for filename in filenames:
            if (len(filenames) - k > count):
                os.remove(outpath + '/' + filename)
                k += 1
            else:
                break
        break
    # os.remove(temppath)


def read_to_pd(mypath):
    i = 1
    df = pd.DataFrame(columns=['subject', 'classname', 'img'])
    logger.info('read class files %s to  DataFrame ...' % mypath)
    file_list = fetch_all_files(mypath)
    for i, file in enumerate(file_list):
        base_dir = os.path.basename(os.path.dirname(file))
        class_num = base_dir.split('-')[0]
        sub_name = cal_file_name(os.path.basename(file))
        df.loc[i] = [sub_name, class_num, file]
    logger.info('read class files finished,len=%s' % i)
    return df


def resize_multi_img_files(src_dir, out_dir, size=(360, 360), n_process=4):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_files = fetch_all_files(src_dir)
    total_files = len(all_files)
    logger.info("start resize_multi_img_files count=%s", total_files)
    length = total_files / float(n_process)
    indices = [int(round(i * length)) for i in range(n_process + 1)]
    sub_list = [all_files[indices[i]:indices[i + 1]] for i in range(n_process)]
    processes = [Process(target=resize_img_files, args=[sub_list[i], out_dir, size]) for i
                 in
                 range(n_process)]

    join_processes(processes)
    logger.info("resize_multi_img_files finished")


def resize_img_files(file_list, out_dir, size=(360, 360)):
    for i in range(len(file_list)):
        file_path = file_list[i]
        img = cv2.resize(cv2.imread(file_path), size)
        out_path = os.path.join(out_dir, os.path.split(file_path)[-1])
        cv2.imwrite(out_path, img)


def load_one_fold_data(src_dir, size=224, class_index=0):
    print('loading train data..')
    imgs = os.listdir(src_dir)
    n = len(imgs)
    x_test = np.zeros((n, size, size, 3), dtype=np.uint8)
    src_files = []
    y_test = []
    for i in range(n):
        file = os.path.join(src_dir, imgs[i])
        src_files.append(file)
        y_test.append(class_index)

        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        x_test[i] = cv2.resize(img, (size, size))

    print('load train data %s ' % (len(x_test)))
    return x_test, y_test, src_files


from multiprocessing import Process, Manager


def load_multi_img_data(src_dir, n_process=2):
    all_files = fetch_all_files(src_dir)
    total_files = len(all_files)
    length = total_files / float(n_process)
    indices = [int(round(i * length)) for i in range(n_process + 1)]
    sub_list = [all_files[indices[i]:indices[i + 1]] for i in range(n_process)]

    with Manager() as manager:
        manage_list = manager.list()  # <-- can be shared between processes.
        processes = [Process(target=load_img_data, args=[sub_list[i], manage_list]) for i
                     in
                     range(n_process)]

        join_processes(processes)
        x_list = []
        src_files_list = []
        logger.debug('manage_list len=%s' % len(manage_list))
        for (X, src_files) in manage_list:
            for i in range(len(src_files)):
                x_list.append(X[i])
                src_files_list.append(src_files[i])
        return np.asarray(x_list), src_files_list


def load_img_data(file_list, manage_list=None, size=(224, 224)):
    print('load_img_data %s..' % len(file_list))

    n = len(file_list)
    data_list = []
    src_files = []
    for i in range(n):
        file = file_list[i]
        img = read_img_file(file)
        if img is None:
            continue

        src_files.append(file)
        img = cv2.resize(img, size)
        data_list.append(img)

    data_list = np.asarray(data_list, dtype=np.uint8)
    if manage_list is not None:
        manage_list.append((data_list, src_files))

    logger.debug("load_img_data finished ,len=%s" % len(data_list))
    return data_list, src_files


def fetch_all_files(from_dir, followlinks=True, file_exts=None, exclude_file_exts=None):
    """
    获取目录下所有文件
    """
    all_files = []
    for root, dirs, files in os.walk(from_dir, followlinks=followlinks):
        for name in files:
            if file_exts:
                _, ext = os.path.splitext(name)
                if ext not in file_exts:
                    logger.debug("exclude file %s,%s" % (name, ext))
                    continue

            if exclude_file_exts:
                _, ext = os.path.splitext(name)
                if ext in exclude_file_exts:
                    logger.debug("exclude file %s,%s" % (name, ext))
                    continue

            path_join = os.path.join(root, name)
            all_files.append(path_join)

    logger.debug("fetch_all_files count=%s" % len(all_files))
    return all_files


def merge_to_one_fold(from_dir, out_dir, n_process=5):
    """
    将多个分类文件合并到一个目录
    """
    all_files = fetch_all_files(from_dir)
    out_dir_list = [out_dir for _ in all_files]

    copy_files_2(all_files, out_dir_list, n_process)


def copy_files_2(all_files, out_dir_list, n_process=5):
    total_files = len(all_files)
    for dist in out_dir_list:
        if not os.path.exists(dist):
            os.makedirs(dist)

    length = total_files / float(n_process)
    indices = [int(round(i * length)) for i in range(n_process + 1)]
    sub_list = [all_files[indices[i]:indices[i + 1]] for i in range(n_process)]
    out_list = [out_dir_list[indices[i]:indices[i + 1]] for i in range(n_process)]
    processes = [Process(target=copy_files, args=[sub_list[i], out_list[i]]) for i
                 in
                 range(n_process)]
    join_processes(processes)


def copy_files(source_files, target_dirs):
    logger.debug("start to  copy files count=%s ..." % len(source_files))
    for i in range(0, len(source_files)):
        dst = target_dirs[i]
        shutil.copy2(source_files[i], dst)


def sort_by_file_name(dirnames):
    if len(dirnames) < 30:
        logger.debug("sort_by_file_name dirs=%s" % dirnames)
    dirs = dirnames
    try:
        dirs = sorted(dirnames, key=lambda x: (int(re.sub('\D', '', x)), x))
    except:
        logger.warn("sort_by_file_name fail")
    return dirs


fine_img_name_len = 30


def cal_file_name(file_name):
    name, file_extension = os.path.splitext(file_name)
    if len(file_name) >= fine_img_name_len:
        sub_name = file_name
    else:

        sub_name = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S_') + str(random.randint(0, 1000000)).zfill(12)
        sub_name += file_extension
        assert len(sub_name) >= fine_img_name_len
    return sub_name


def fix_files(respath, fixpath):
    """
    标错了，更正过来
    """
    df = read_to_pd(respath)
    df2 = read_to_pd(fixpath)
    print(df2.head(10))
    has_fixed = []
    for name in df['subject']:
        find = df2[df2['subject'] == name]
        if (len(find.values) > 0):
            old_class = df[df['subject'] == name].values[0][1]
            clasname = find.values[0][1]
            if (old_class == clasname):
                continue
            print(clasname)
            file_path = df[df['subject'] == name].values[0][2]
            paths = file_path.split('/')
            mv = paths[:-2] + [str(clasname)] + paths[-1:]
            dest = '/'.join(mv)
            print(dest)

            try:
                shutil.move(file_path, dest)
                has_fixed.append(dest)
            except:
                traceback.print_exc()
    logger.info('fixed files  ' + str(len(has_fixed)))


def choice_exam(src_dir, out_dir, count):
    class_count = len(os.listdir(src_dir))
    df = read_to_pd(src_dir)
    for i in range(0, class_count):
        values = df[df['classname'] == str(i)].values
        r = np.random.choice(len(values), count)
        files = [values[x][2] for x in r]
        target_dir = out_dir + '/' + str(i)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for f in files:
            if os.path.exists(f):
                shutil.move(f, target_dir)


def read_img_from_base64(request_data, param_key='imgBase64'):
    if not request_data:
        logger.warning("can't find request_data ")
        return None
    img = request_data.get(param_key, None)
    if not img:
        logger.warning("can't find imgBase64 from data")
        return None
    byte_data = img.split(',')[-1]
    file_bytes = base64.decodebytes(bytes(byte_data, "utf-8"))
    np_arr = np.fromstring(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_params = request_data.get('img_params', None)
    if img_params:
        hue_rotate = img_params.get('hue_rotate', 0)
        saturate = img_params.get('saturate', 1)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float64")
        # hue_rotate：scale(0,180)，终端：scale(-180,+180)
        img_hsv[:, :, :1] += hue_rotate / 2
        # saturate：scale(0,255)，终端：scale(-100,+100)
        img_hsv[:, :, 1:2] += saturate * 255 / 200
        img_hsv[np.where(img_hsv < 0)] = 0
        img_hsv[np.where(img_hsv >= 255)] = 255
        img = cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img


def encode_base64_img(byte_data):
    return str(base64.b64encode(byte_data)).replace("b'", 'data:image/png;base64,').replace("'", "")


def read_img_from_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        np_arr = np.fromstring(r.content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    else:
        logger.warn("read img from url  fail %s" % url)
    return None


def get_video_fps(video_capture):
    major_ver, _, _ = (cv2.__version__).split('.')
    return video_capture.get(cv2.cv.CV_CAP_PROP_FPS if int(major_ver) < 3 else cv2.CAP_PROP_FPS)


# frame_per_sec: 0表示截所有视频帧
def split_video(filepath, frame_per_sec=5, limit=None):
    logger.debug("split video file:" + filepath)
    filepath = os.path.abspath(filepath)

    video_capture = cv2.VideoCapture(filepath)
    fps = get_video_fps(video_capture)
    if frame_per_sec:
        step = int(round(fps / frame_per_sec, 0))
    else:
        step = 1

    count = 0
    success = True
    img_files = []
    if not limit:
        limit = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    while success and len(img_files) < limit:
        success, image = video_capture.read()
        if not success:
            break
        if count % step == 0:
            img_files.append(image)
        count += 1

    print('spit fished ,total imgs=%s' % len(img_files))
    video_capture.release()
    return img_files,step


def save_cache(data, file_path):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()
    logger.debug("save cache to file %s" % file_path)


def load_cache(file_path):
    data = dict()
    if os.path.isfile(file_path):
        file = open(file_path, 'rb')
        data = pickle.load(file)
        file.close()
        logger.debug("load cache from file %s" % file_path)
    else:
        logger.debug('cache file not exist: ' + file_path)
    return data


class CacheFold:
    def __init__(self, path, max_day=90):
        self.path = path
        self.max_day = max_day

    def start_monitor(self):

        pro = Process(target=self._start_monitor)
        pro.start()

    def _start_monitor(self):
        logger.debug("monitor cache fold %s ,max days=%s" % (self.path, self.max_day))
        while True:
            try:
                self.clean_dir()
            except:
                traceback.print_exc()
                logger.error("update data err")
            time.sleep(60 * 60 * 4)

    def clean_dir(self):
        if not os.path.exists(self.path):
            return
        dirs = os.listdir(self.path)
        now = datetime.datetime.now()
        for fold in dirs:
            try:
                create_at = datetime.datetime.strptime(fold, "%Y-%m-%d")
                if now.timestamp() - create_at.timestamp() > self.max_day * 24 * 3600:
                    fold_path = os.path.join(self.path, fold)
                    shutil.rmtree(fold_path)
                    logger.info("del out of date fold %s" % fold_path)
            except:
                logger.info('ignore file %s ' % fold)


def Hamming_distance(hash_target, hash_match):
    num = 0
    if len(hash_target) != len(hash_match) or len(hash_target) * len(hash_match) == 0:
        return -1
    for index in range(len(hash_target)):
        if hash_target[index] != hash_match[index]:
            num += 1
    return num


def resize_image(img, size):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    img = np.array(new_image)
    return img[:, :, ::-1]


def dhash(img_src, hash_size=8):
    img = cv2.resize(img_src, (360, 360))
    img = img[30:330, 30:330]
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_RGB)
    # Grayscale and shrink the image in one step.
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
    #     pixels = list(image.getdata())
    # Compare adjacent pixels.
    hash = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            if pixel_left > pixel_right:
                hash.append(1)
            else:
                hash.append(0)
                #     print(len(hash))
    return hash


class JoinVideo(object):
    def __init__(self, img, result):
        self.img_src = img
        self.result = result

    def cal_img_dhash(self, hash_size=8):
        image = cv2.resize(self.img_src, (360, 360))
        _image = image[30:330, 30:330]
        img_RGB = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_RGB)
        # Grayscale and shrink the image in one step.
        img = img.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
        # Compare adjacent pixels.
        hash_code = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = img.getpixel((col, row))
                pixel_right = img.getpixel((col + 1, row))
                if pixel_left > pixel_right:
                    hash_code.append(1)
                else:
                    hash_code.append(0)
        return hash_code

    def string_hash(self, hash_code):
        string_hash = ""
        for code in hash_code:
            string_hash += str(code)
        return string_hash

    def add_hash_result(self):
        hash_code = self.string_hash(self.cal_img_dhash())
        label_result = LabelResult("哈希指纹")
        label_result.class_name = hash_code
        self.result.add(label_result)
        return self.result


def draw_cam(model, index, last_conv_layer_name, last_conv_layer_out_channel):
    # This is the "african elephant" entry in the prediction vector
    #     african_elephant_output = model.output[:, index]
    african_elephant_output = model.output[:, index]
    # The is the output feature map of the `block5_conv3` layer,
    # the last convolutional layer in VGG16
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # This is the gradient of the "african elephant" class with regard to
    # the output feature map of `block5_conv3`
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

    # This is a vector of shape (512,), where each entry
    # is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # This function allows us to access the values of the quantities we just defined:
    # `pooled_grads` and the output feature map of `block5_conv3`,
    # given a sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # These are the values of these two quantities, as Numpy arrays,
    # given our sample image of two elephants
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the elephant class
    for i in range(last_conv_layer_out_channel):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    return heatmap


def cv2FindContours(image):
    if cv2.__version__[0] == '4':
        ctrs, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, ctrs, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return ctrs
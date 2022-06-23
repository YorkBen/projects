import os
import sys

# os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
print(tf.test.is_gpu_available())

sys.path.insert(0, '../Lib/trainer')
from mbsh.core.unet_pp.segmentation_models import Unet, Xnet
# from mbsh.core.images import fetch_all_files, split_video
from mbsh.core.xnet import mini_unet_model,xnet_predict_file

from crop_func import process_path


model_path = r'models/X_ca_region_202205191033_35-0.993.hdf5'

img_size = (512, 512)
input_size = img_size + (3,)
use_unet = False
# 视频图转换后图像大小
# 不改变尺寸则设置为None
img_resize = None #(512, 512)

# 创建模型并加载
if use_unet:
#     model = unet(input_size=input_size)
    model = Unet(input_shape=input_size, backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose')
else:
    model = Xnet(input_shape=input_size, backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose')
model.load_weights(model_path)


in_path = r'data\胆结石分割模型'
out_path = in_path + '_out'
os.makedirs(out_path, exist_ok=True)

subs = os.listdir(in_path)
for sub in subs:
    if 'mask' in sub:
        imgs_path = os.path.join(in_path, sub[:-5])
        labels_path = os.path.join(in_path, sub)

        out_imgs_path = os.path.join(out_path, 'images')
        out_labels_path = os.path.join(out_path, 'labels')
        if not os.path.exists(out_imgs_path):
            os.makedirs(out_imgs_path, exist_ok=True)
        if not os.path.exists(out_labels_path):
            os.makedirs(out_labels_path, exist_ok=True)

        process_path(model, imgs_path, labels_path, out_imgs_path, out_labels_path, img_size)

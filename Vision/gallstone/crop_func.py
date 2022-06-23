import os
import sys
from imageio import imread,imwrite
from PIL import Image
import numpy as np
import shutil
import cv2

sys.path.insert(0, '../Lib/trainer')
from mbsh.core.unet.data import *

img_resize = None #(512, 512)

# 根据轮廓点列表计算轮廓面积
def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])
    return area


def takeLength(elem):
    return areaCal([elem])

def resize_image(img, size):
    """
    resize 保持原图的长宽比
    """
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     print(image.size,image.shape,type(image))
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    img = np.array(new_image)
    return (scale, nw, nh), img[:, :, ::-1]

def process_img(model, src_img, img_size):
    src_img_bak = src_img.copy()
    src_h = src_img_bak.shape[0]
    src_w = src_img_bak.shape[1]

    new_w, new_h = img_size

    (scale, nw, nh), img = resize_image(src_img, img_size)
    print((scale, nw, nh))

    # trans.resize把img从numpy.uint8类型转换为numpy.float64
    img = trans.resize(img, img_size)

    # reshape把img从(512, 512, 3)转换为(1, 512, 512, 3)
    img = np.reshape(img, (1,) + img.shape)

    # 模型识别图片, results为(1, 512, 512, 1)，是灰度图片
    results = model.predict(np.array(img),verbose=1)
    print(results.shape)

    # gray_img是（512，512）的灰度图，元素类型numpy.float32
    gray_img = results[0][:,:,0]
    gray_img[np.where(gray_img <= 0.3)] = 0

    # heatmap转为灰度图
    bw_heatmap = np.uint8(255 * gray_img)
    bw_heatmap[bw_heatmap != 0] = 255
    if cv2.__version__[0] == '4':
        ai_ctrs, _ = cv2.findContours(bw_heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, ai_ctrs, _ = cv2.findContours(bw_heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(ai_ctrs) == 0:
        print('len ai ctrs 000000000000')
        return False,src_img

#     print(ai_ctrs, 'ppppppp')
    ai_ctrs = list(ai_ctrs)
    ai_ctrs.sort(key=takeLength, reverse=True)


    x,y,w,h = cv2.boundingRect(ai_ctrs[0])

    # 映射到原图尺寸的坐标系
    x = x - (new_w - nw) // 2
    y = y - (new_h - nh) // 2

    new_x = 0 if x < 0 else x
    new_y = 0 if y < 0 else y

    x1 = int(new_x / scale)
    y1 = int(new_y / scale)
    w1 = int(w / scale)
    h1 = int(h / scale)

    img_out = src_img_bak[y1:y1+h1, x1:x1+w1, :]
#     print(x1, y1, w1, h1, '3333333333')
    return True,img_out, (x1, y1, w1, h1)
#     return True,img_out[: , : , : : -1], (x1, y1, w1, h1)

def predict_file(model, file, test_label_path=None):
    file_path,file_name = os.path.split(file)
    src_img,reg_ai_ctrs = xnet_predict_file(model=model, file=file, model_input_img_size=img_size, predict_area_num=2,
                                       min_area=0.001)

    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]

    black_mask_image = Image.new('RGB', (src_img_w, src_img_h), (0, 0, 0))
    black_mask_image = np.array(black_mask_image)

    last_ai_mask_image = black_mask_image.copy()
    last_ai_area = 0

#     for item in reg_ai_ctrs:
#         # 轮廓点稀疏化
#         sparse_ai_ctrs = sparse_ctrs([item], gap=points_gap)

#         # (78, 2)
#     #     print(sparse_ai_ctrs[0].shape)
#         sparse_ai_ctrs = np.expand_dims(sparse_ai_ctrs, axis=2)
#         # (78, 1, 2)

#         ai_mask_image = black_mask_image.copy()
#         cv2.fillPoly(ai_mask_image, sparse_ai_ctrs, (255, 255, 255))

#         # 识别得到的黑白图
#     #     save_img_file(os.path.join(result_path, file_name[:-4]+'_aimask'+file_name[-4:]), ai_mask_image)
#         ai_area = black_white_statistics(ai_mask_image, 'ai mask')

#         last_ai_area += ai_area
#         last_ai_mask_image += ai_mask_image

    cv2.drawContours(src_img, reg_ai_ctrs, -1, color=reg_color, thickness=2)
    save_img_file(os.path.join(r'data/胆结石',file_name),src_img)


def process_path(model, imgs_path, labels_path, out_imgs_path, out_labels_path, img_size):
    files = os.listdir(imgs_path)
    for file in files:
        ext = file.split('.')[-1]
        if ext == 'csv':
            continue

        abs_file = os.path.join(imgs_path, file)
#         predict_file(model, abs_file)
#         continue
#         src_img = imread(abs_file)
        try:
            src_img = imread(abs_file)
        except:
            print('read error file: ', abs_file)
            return

#         print(src_img.shape)
        if src_img.shape[1] / src_img.shape[0] > 1.2 or src_img.shape[0] / src_img.shape[1] > 1.2:
            result, out_img, rect = process_img(model, src_img, img_size)
#             print(rect, 'kkkkkkkk')
        else:
#             continue
            result, out_img, rect = 1,src_img,None #[0,0,-1,-1]
#             result, out_img, rect = 1,src_img[: , : , : : -1],None #[0,0,-1,-1]
        if not result:
#             print('detect rect error file: ', abs_file)
            return

        if img_resize:
            _,out_img = resize_image(out_img, img_resize)

        # 处理label
        img_fn = os.path.join(labels_path, file[:])
        if not os.path.exists(img_fn):
            continue

        new_fn = os.path.join(out_imgs_path, file)
#         cv2.imencode('.png', out_img[: , : , : : -1])[1].tofile(new_fn)
        cv2.imencode('.png', out_img)[1].tofile(new_fn)


        img = np.array(Image.open(img_fn))
        if rect:
            img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        if img_resize:
            _,img = resize_image(img, img_resize)

        new_fn = os.path.join(out_labels_path, file)
        img = Image.fromarray(img)
        img.save(new_fn)


# def process_files(path, target_dir, img_size):
#     img_fn = os.path.join(path,'img.png')
#     try:
#         src_img = imread(img_fn)
#     except:
#         print('read error file: ', img_fn)
#         return

#     result, out_img, rect = process_img(src_img, img_size)
#     if not result:
#         print('detect rect error file: ', img_fn)
#         return

#     if img_resize:
#         print(out_img.shape)
#         _,out_img = resize_image(out_img, img_resize)

# #     fpath,fname = os.path.split(file)
#     new_fn = os.path.join(target_dir, 'img_crop.png')
#     cv2.imencode('.png', out_img[: , : , : : -1])[1].tofile(new_fn)


#     img_fn = os.path.join(path,'label.png')
# #     img = Image.open(img_fn)

#     img = np.array(Image.open(img_fn))
#     img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
# #     print(img.shape,'ttt')

#     if img_resize:
#         _,img = resize_image(img, img_resize)

#     img = Image.fromarray(img, mode='P')
#     colormap = label_colormap(255)
#     img.putpalette((colormap * 255).astype(np.uint8).flatten())


#     new_fn = os.path.join(target_dir, 'label_crop.png')
#     img.save(new_fn)

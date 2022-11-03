# encoding: utf-8
"""
@File   : slide_cut.py
@Time   : 2022/6/14 10:59
@author : zk
@license: @EndoAngel
@Desc   :  
"""
import os
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 14330458600
# 使用 Open Slide 打开 mrxs
img_path = r"E:\projects\znyx-trainer\books_2022\04-基础AI\prostate\TCGA-3L-AA1B-01A-01-TS1.9C415218-D5B4-4945-B243-F42A4C8C0484.svs"
slide = openslide.open_slide(img_path)

# 按照 3000 x 3000 切割图片，如果想要更大的图，可以自己调整，overlap 是指图片边界不重叠
data_gen = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)
# 拿到最高像素的缩放级别
level = data_gen.level_count - 1

# 获取最高像素下分割后的列数和行数
[cols, rows] = data_gen.level_tiles[-1]
print(cols, rows)
# 循环读取各个行列的图像
for col in range(cols):
    for row in range(rows):
        print("Processing %s %s" % (col, row))
        im = data_gen.get_tile(level, (col, row))
        w, h = im.size
        if w == 224 and h == 224:
            img_np = np.asarray(im)
            img_ava = np.mean(img_np, axis=2)
            img_ava_bool = img_ava[img_ava < 220]
            if np.sum((img_ava < 220).astype(int)) > 224 * 224 * 0.5:
            # print((img_ava < 220).astype(int))
            # if img_np < 220:
              im.save("E:\\projects\\znyx-trainer\\books_2022\\04-基础AI\\prostate\\新建文件夹\\%s_%s.jpg" % (col, row))


    # extrema = im.convert("L").getextrema()
    # if extrema[0] != extrema[1]:
    #   # 保存图像到 output 文件夹（需要提前创建这个文件夹）
    #   im.save("E:\\projects\\znyx-trainer\\books_2022\\04-基础AI\\prostate\\新建文件夹\\%s_%s.jpg" % (col, row))

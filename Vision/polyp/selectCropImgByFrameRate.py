import os
import math
import shutil

base_path = r'data\input\imgs'
def select_process(dirname, srcFrameRate, targetFrameRate=20):
    """
    从裁剪的视频图片中按照目标帧率来选择图片
    """
    print('processing %s' % dirname)
    srcdir = os.path.join(base_path, dirname)
    targetdir = os.path.join(base_path, dirname + '_select')
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    imgs = os.listdir(srcdir)
    imgs_no = sorted([int(i.replace('.jpg', '')) for i in imgs])
    # imgs_sorted = [str(img) + '.jpg' for img in imgs_no]
    imgs_no_ = [int(math.floor(no * targetFrameRate / srcFrameRate)) for no in imgs_no]
    imgs_choosed = [e for e in imgs_no_]
    for i in range(len(imgs_no_)-1, 0, -1):
        imgs_choosed[i] = imgs_no_[i] - imgs_no_[i-1]

    # print(imgs_no[:30])
    # print(imgs_no_choosed[:30])
    for a,b in zip(imgs_no, imgs_choosed):
        if b == 1:
            img_name = str(a) + '.jpg'
            if not os.path.exists(os.path.join(targetdir, img_name)):
                shutil.copyfile(os.path.join(srcdir, img_name), os.path.join(targetdir, img_name))

# 帧率
# 2008 -> 30
# 2010 -> 30
# 2016 -> 30
# 2019 -> 30
# 3006 -> 48
# 3450 -> 20
# 3452 -> 20
# 3456 -> 49
# 3466 -> 50
# 3486 -> 50
# select_process('2008_crop', 30, 20)
# select_process('2010_crop', 30, 20)
# select_process('2016_crop', 30, 20)
select_process('2019_crop', 30, 20)
# select_process('3006_crop', 48, 20)
# select_process('3450_crop', 20, 20)
# select_process('3452_crop', 20, 20)
# select_process('3456_crop', 49, 20)
# select_process('3466_crop', 50, 20)
# select_process('3486_crop', 50, 20)

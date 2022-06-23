import os
import cv2


img_folder = r'3450'
input_img_path = os.path.join(r'data\input\imgs', img_folder + '_crop')
output_img_path = os.path.join(r'data\output\imgs', img_folder)
# output_video_path = os.path.join(r'data\output\video', img_folder + '.mp4')

fps = 20
delta = 4 # 20 * 0.2

files = os.listdir(output_img_path)
names = sorted([int(file.split('.')[0]) for file in files])
# sorted_files = [str(n) + '.jpg' for n in names]

# 格式： start, end
polyps = []
start, last, end, frame_ct = names[0], names[0], -1, 1
for name in names[1]:
    if name - last > 4:
        # 到了截断处
        end = last
        if frame_ct >= 4:
            polyps.append((start, end))

        # 初始化
        start, last, end, frame_ct = names, names, -1, 1
    else:
        frame_ct = frame_ct + 1
        last = name

print('假阳性个数：%s, 假阳性个数/S：%s' % (len(polyps), len(polyps) / (len(os.listdir(input_img_path)) // 20)))

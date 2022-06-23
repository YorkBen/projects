import os
import cv2
import time

img_folder = r'3450'
input_img_path = os.path.join(r'data\input\imgs', img_folder + '_crop')
output_img_path = os.path.join(r'data\output\imgs', img_folder)
output_video_path = os.path.join(r'data\output\video', img_folder + '.avi')

files = os.listdir(input_img_path)
img = cv2.imread(os.path.join(input_img_path, files[0]))
size = (img.shape[1], img.shape[0])

fps = 20
# video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
videoWrite = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

files = os.listdir(input_img_path)
for no in range(1, len(files) + 1):
    filename = str(no) + '.jpg'
    output_file = os.path.join(output_img_path, filename)
    input_file = os.path.join(input_img_path, filename)
    if os.path.exists(output_file):
        img_path = output_file
    else:
        img_path = input_file

    img = cv2.imread(img_path)
    # img = cv2.resize(img_path,  (604, 604))#resize图片
    videoWrite.write(img)

videoWrite.release()
# cv2.destroyAllWindows()

import cv2
import os
import sys
import numpy as np

def video_to_photo(video_path, video_img_folder, target_fps=0):
    # create video img folder
    if not os.path.exists(video_img_folder):
        os.makedirs(video_img_folder)

    # read video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    choose_rate = target_fps / fps
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print('processing video: %s' % video_path)
    print('video frame num: %s' % total_frame_num)
    print('video fps: %s, choose fps: %s, choose rate: %.2f' % (fps, target_fps, choose_rate))
    print('frame should choose: %d' % (choose_rate * total_frame_num))

    # The number of videos cropped from the video
    # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # whether it is opened normally
    total_ct, choose_ct = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if ret == True:
            total_ct = total_ct + 1
            if int(total_ct * choose_rate) - choose_ct == 1:
                choose_ct = choose_ct + 1
                # if count <= num_frames:
                save_root = os.path.join(video_img_folder, str(choose_ct) + '.jpg')
                # print("writing " + save_root + " ...")
                cv2.imwrite(save_root, frame)
                if choose_ct > 0 and (choose_ct % 100) == 0:
                    print('frame choosed: %s' % choose_ct)
        cv2.waitKey(1)
    cap.release()


def cropImg(video_img_folder, crop_img_folder, region, debug=False):
    # create crop img folder
    if not os.path.exists(crop_img_folder):
        os.makedirs(crop_img_folder)

    for file in os.listdir(video_img_folder):
        img = cv2.imdecode(np.fromfile(os.path.join(video_img_folder, file), dtype=np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.imread()
        img_ = img[region[0]:region[1], region[2]:region[3]]

        if debug:
            cv2.namedWindow("enhanced", 0);
            cv2.imshow('enhanced', img_)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()

        cv2.imwrite(os.path.join(crop_img_folder, file), img_)



if __name__ == '__main__':
    # base_dir = r'D:\项目资料\息肉假阳性\20221007'
    base_dir = r'/mnt/d/项目资料/息肉假阳性/20221007'

    video_folder = os.path.join(base_dir, 'videos_wfz')
    img_folder = os.path.join(base_dir, 'images_wfz')
    crop_img_folder = os.path.join(base_dir, 'images_crop_wfz')
    target_fps = 18
    # for filename in os.listdir(video_folder):
    regions = [
        [30,1045, 696,1860],
        [0,950, 465,1560],
        [0, 1100, 574, 1836],
        [0, 1100, 664, 1946],
        [0, 1100, 578, 1826],
        [34, 1044, 657, 1826],
        [34, 1044, 727, 1642],
        [0, 1100, 662, 1642],
        [30,1045, 656,1830]
    ]
    # videos = {'1006-原视频.mp4':0}
    # 1037, 1067, 1087, 1112, 1156, 1255, 1264, 1293, 1307, 1344, 1435, 2003
    #
    # videos = {name:0 for name in os.listdir(video_folder)}
    # print(videos)
    videos = {
    '1006-原视频.mp4': 0,
    '1014-原视频.mp4': 0,
    '1037-原视频.mp4': 2,
    '1067-原视频.mp4': 2,
    '1087-原视频.mp4': 2,
    '1112-原视频.mp4': 3,
    '1149-原视频.mp4': 3,
    '1156-原视频.mp4': 4,
    '1162-原视频.mp4': 6,
    '1202-原视频.mp4': 7,
    '1255-原视频.mp4': 3,
    '1264-原视频.mp4': 3,
    '1293-原视频.mp4': 3,
    '1307-原视频.mp4': 8,
    '1354-原视频.mp4': 0,
    '1435-原视频.mp4': 8,
    '1447-原视频.mp4': 0,
    '2003-原视频.mp4': 1,
    '2135-原视频.mp4': 3,
    '2154-原视频.mp4': 3,
    '2156-原视频.mp4': 0,
    '2161-原视频.mp4': 0,
    '2195-原视频.mp4': 3,
    '2230-原视频.mp4': 0,
    '3398-原视频.mp4': 0,
    '3399-原视频.mp4': 0,
    '3402-原视频.mp4': 0,
    '3414-原视频.mp4': 0,
    '3415-原视频.mp4': 0,
    '3416-原视频.mp4': 0,
    '3419-原视频.mp4': 0,
    '3429-原视频.mp4': 0,
    '3434-原视频.mp4': 0,
    '3435-原视频.mp4': 0,
    '3438-原视频.mp4': 0,
    '3442-原视频.mp4': 0,
    '3479-原视频.mp4': 0,
    '3497-原视频.mp4': 0,
    '3513-原视频.mp4': 0,
    '3538-原视频.mp4': 0}


    for filename in list(videos.keys()):
        print('processing: %s' % filename)
        video_path = os.path.join(video_folder, filename)
        name = filename.replace('-原视频', '').replace('.mp4', '')
        video_img_folder = os.path.join(img_folder, name)
        # video_to_photo(video_path, video_img_folder, target_fps)

        video_crop_img_folder = os.path.join(crop_img_folder, name)
        region = regions[videos[filename]]
        cropImg(video_img_folder, video_crop_img_folder, region, False)


##

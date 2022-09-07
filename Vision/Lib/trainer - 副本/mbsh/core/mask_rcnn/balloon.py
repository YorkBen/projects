"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import csv
import pandas as pd

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
# COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"


# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
# print(ROOT_DIR)

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library

#
# # Directory to save logs and model checkpoints, if not provided
# # through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "diff"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):
    #
    # def load_balloon(self, dataset_dir, subset):
    #     """Load a subset of the Balloon dataset.
    #     dataset_dir: Root directory of the dataset.
    #     subset: Subset to load: train or val
    #     """
    #     # Add classes. We have only one class to add.
    #     self.add_class("diff", 1, "diff")
    #
    #     # Train or validation dataset?
    #     assert subset in ["train", "val"]
    #     dataset_dir = os.path.join(dataset_dir, subset)
    #
    #     # Load annotations
    #     # VGG Image Annotator (up to version 1.6) saves each image in the form:
    #     # { 'filename': '28503151_5b5b7ec140_b.jpg',
    #     #   'regions': {
    #     #       '0': {
    #     #           'region_attributes': {},
    #     #           'shape_attributes': {
    #     #               'all_points_x': [...],
    #     #               'all_points_y': [...],
    #     #               'name': 'polygon'}},
    #     #       ... more regions ...
    #     #   },
    #     #   'size': 100202
    #     # }
    #     # We mostly care about the x and y coordinates of each region
    #     # Note: In VIA 2.0, regions was changed from a dict to a list.
    #     annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    #     annotations = list(annotations.values())  # don't need the dict keys
    #
    #     # The VIA tool saves images in the JSON even if they don't have any
    #     # annotations. Skip unannotated images.
    #     annotations = [a for a in annotations if a['regions']]
    #
    #     # Add images
    #     for a in annotations:
    #         # Get the x, y coordinaets of points of the polygons that make up
    #         # the outline of each object instance. These are stores in the
    #         # shape_attributes (see json format above)
    #         # The if condition is needed to support VIA versions 1.x and 2.x.
    #         if type(a['regions']) is dict:
    #             polygons = [r['shape_attributes'] for r in a['regions'].values()]
    #         else:
    #             polygons = [r['shape_attributes'] for r in a['regions']]
    #
    #         # load_mask() needs the image size to convert polygons to masks.
    #         # Unfortunately, VIA doesn't include it in JSON, so we must read
    #         # the image. This is only managable since the dataset is tiny.
    #         image_path = os.path.join(dataset_dir, a['filename'])
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #         height, width = image.shape[:2]
    #
    #         self.add_image(
    #             "balloon",
    #             image_id=a['filename'],  # use file name as a unique image id
    #             path=image_path,
    #             width=width, height=height,
    #             polygons=polygons)

    # for csv
    def load_data_from_csv_multi(self, dataset_dir, csv_name='via_region_data.csv', source='region',
                                 label_dict=None, image_files=None):
        # self.add_class(class_name, 1, class_name)
        img_filename_list = []
        self.label_dict = label_dict

        df, fn_key = get_data_from_csv(os.path.join(dataset_dir, csv_name))
        filename_col = df.drop_duplicates(fn_key)

        for i, row in enumerate(filename_col.values):
            if row[5] == '{}':
                continue

            img_filename = row[0]
            image_path = os.path.join(dataset_dir, img_filename)
            if not os.path.exists(image_path):
                print('file not exists: ', image_path)
                continue

            if image_files:
                if img_filename not in image_files:
                    continue

            bl_df = df[df[fn_key] == img_filename]
            polygons = []
            for j in range(len(bl_df)):
                dict1 = json.loads(bl_df.iloc[j,5])
                if len(dict1) == 0:
                    continue

                dict2 = {}
                if len(label_dict) > 2:
                    dict2 = json.loads(bl_df.iloc[j,6])
                    if len(dict2) == 0:
                        # 无类型标签
                        continue

                polygons.append(dict(dict1, **dict2))

            with open(image_path, 'rb') as img_file:
                _bytes = img_file.read()
                nparr = np.fromstring(_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]

            img_filename_list.append(img_filename)
            self.add_image(
                source,
                image_id=img_filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

        print(self.label_dict)


    # for csv
    def load_data_from_csv(self, dataset_dir, csv_name='via_region_data.csv', class_name='region', image_files=None):
        # Add classes. We have only one class to add.
        self.add_class(class_name, 1, class_name)
        img_filename_list = []

        df, fn_key = get_data_from_csv(os.path.join(dataset_dir, csv_name))
        filename_col = df.drop_duplicates(fn_key)

        for i, row in enumerate(filename_col.values):
            img_filename = row[0]
            image_path = os.path.join(dataset_dir, img_filename)
            if not os.path.exists(image_path):
                print('file not exists: ', image_path)
                continue

            if image_files:
                if img_filename not in image_files:
                    continue

            bl_df = df[df[fn_key] == img_filename]
            polygons = []
            for j in range(len(bl_df)):
                tmp_dict = json.loads(bl_df.iloc[j,5])
                if len(tmp_dict):
                    polygons.append(tmp_dict)

            with open(image_path, 'rb') as img_file:
                _bytes = img_file.read()
                nparr = np.fromstring(_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[:2]

            img_filename_list.append(img_filename)
            self.add_image(
                class_name,
                image_id=img_filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)


    # for csv
    def load_data_from_csv_old(self, dataset_dir, csv_name='via_region_data.csv', class_name='region', image_files=None):
        # Add classes. We have only one class to add.
        self.add_class(class_name, 1, class_name)

        image_list_path = os.path.join(dataset_dir, csv_name)

        img_filename_list = []

        with open(image_list_path, encoding='UTF-8') as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                img_filename = row[0]
                if img_filename in img_filename_list:
                    continue

                polygons = [json.loads(row[5])]
                if image_files:
                    if img_filename not in image_files:
                        continue

                image_path = os.path.join(dataset_dir, img_filename)

                with open(image_path, 'rb') as img_file:
                    _bytes = img_file.read()
                    nparr = np.fromstring(_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                height, width = image.shape[:2]

                img_filename_list.append(img_filename)
                self.add_image(
                    class_name,
                    image_id=img_filename,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)


    def load_mask_multi(self, image_id, multi=True):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        # image_info = self.image_info[image_id]
        # if image_info["source"] != self.class_info[0]["name"]:
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        if len(info["polygons"]) == 0:
            return None

        mask = np.zeros([info["height"], info["width"], 3], dtype=np.uint8)

        new_dict = {v: k for k, v in self.label_dict.items()}

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'polygon' or p['name'] == 'polyline':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            elif p['name'] == 'ellipse':
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
            else:
                rx = int(p['x'])
                ry = int(p['y'])
                rw = int(p['width'])
                rh = int(p['height'])
                rect_x = [rx,(rx + rw),(rx + rw),rx]
                rect_y = [ry,ry,(ry + rh),(ry + rh)]
                rr, cc = skimage.draw.polygon(rect_y, rect_x)

            mask[rr, cc] = new_dict[p['label']] if multi else 1

        return mask


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != self.class_info[1]["name"]:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'polygon' or p['name'] == 'polyline':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
            elif p['name'] == 'ellipse':
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
                mask[rr, cc, i] = 1
            else:
                rx = int(p['x'])
                ry = int(p['y'])
                rw = int(p['width'])
                rh = int(p['height'])
                rect_x = [rx,(rx + rw),(rx + rw),rx]
                rect_y = [ry,ry,(ry + rh),(ry + rh)]
                rr, cc = skimage.draw.polygon(rect_y, rect_x)
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    #
    # def image_reference(self, image_id):
    #     """Return the path of the image."""
    #     info = self.image_info[image_id]
    #     if info["source"] == "no_diff":
    #         return info["path"]
    #     else:
    #         super(self.__class__, self).image_reference(image_id)


def get_black_white_mask_img(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make black and white copy of the image. They still has 3 RGB channels, though.
    black = np.zeros([image.shape[0], image.shape[1], 3], np.uint8)
    white = black + 255

    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, white, black).astype(np.uint8)
    else:
        splash = black.astype(np.uint8)

    return splash


def get_data_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', engine='python')
    except:
        df = pd.read_csv(csv_file, encoding='gbk', engine='python')

    try:
        tmp = df['filename']
        fn_key = 'filename'
    except:
        fn_key = '#filename'

    return df, fn_key
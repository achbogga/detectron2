#!/usr/bin/env python
# coding: utf-8

# # install packages

# In[1]:

# !python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

# In[2]:

# !pip install ipywidgets

# In[3]:

# !pip install opencv-python

# In[4]:

# !pip install imutils

# In[5]:

# !pip install albumentations


# In[6]:


# !pip install albumentations==0.4.6


# In[7]:


# !python3 -m pip install pandas


# # import

# In[8]:


import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.data import MetadataCatalog,DatasetCatalog

from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from detectron2.data.datasets import register_coco_instances,load_coco_json

import json

import copy

import pandas as pd

from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
import copy

from imutils import paths
import argparse
import cv2
import tqdm

#import ipywidgets as widgets
#from ipywidgets import interact, interact_manual


# # model

# In[9]:


DatasetCatalog.clear()


# In[10]:


#change paths for your data (images folder and annotations.json)


# In[11]:


for d in ['train','test']:
    DatasetCatalog.register("octiva_"+d, lambda d=d: load_coco_json("/home/aboggaram/data/Octiva/consolidated_coco_format_validated_11_01_2022/{}.json".format(d,d),
    image_root= "/home/aboggaram/data/Octiva/data_for_playment",\
    dataset_name="octiva_"+d))


# In[12]:


dataset_dicts_train = DatasetCatalog.get("octiva_train")


# In[13]:


dataset_dicts_test = DatasetCatalog.get("octiva_test")


# In[14]:


train_metadata = MetadataCatalog.get("octiva_train")

test_metadata = MetadataCatalog.get("octiva_test")


# In[15]:


# import os
# from IPython.display import Image
# example = dataset_dicts_train[0]
# image = utils.read_image(example["file_name"], format="RGB")
# print(example)
# plt.figure(figsize=(5,5),dpi=200)
# visualizer = Visualizer(image[:, :, ::-1], metadata=train_metadata, scale=1.0)
# vis = visualizer.draw_dataset_dict(example)
# plt.imshow(vis.get_image()[:, :,::-1])
# plt.show()


# # copy-paste

# In[16]:


#import scripts
from copy_paste import CopyPaste
from coco import CocoDetectionCP


# In[17]:


import albumentations as A


# In[18]:


aug_list = [A.Resize(800,800),#resize all images to fixed shape
        CopyPaste(blend=True, sigma=1, pct_objects_paste=0.9, p=1.0) #pct_objects_paste is a guess
    ]
        
    
#you can add any augmentation from albumentations to this list, for example, you can use:


aug_list = [A.Resize(800,800),\
            A.OneOf([A.HorizontalFlip(),A.RandomRotate90()],p=0.75),\
            A.OneOf([A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),A.RandomGamma(),A.CLAHE()],p=0.5),\
            A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15)],p=0.5),\
            A.OneOf([A.Blur(),A.MotionBlur(),A.GaussNoise(),A.ImageCompression(quality_lower=75)],p=0.5),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=0.9, p=1.0) #pct_objects_paste is a guess
    ]


transform = A.Compose(
            aug_list, bbox_params=A.BboxParams(format="coco")
        )


# add path to your images and annotations

# In[19]:


data = CocoDetectionCP(
    '/home/aboggaram/data/Octiva/data_for_playment', 
    '/home/aboggaram/data/Octiva/consolidated_coco_format_validated_11_01_2022/train.json', 
    transform
)


# # visualize

# In[20]:


from visualize import display_instances


# augmentation change after each iteration

# In[21]:


# for i in range(5):

#     img_data = data[0]

#     f, ax = plt.subplots(1, 2, figsize=(16, 16))
#     image = img_data['image']
#     masks = img_data['masks']
#     bboxes = img_data['bboxes']

#     empty = np.array([])
#     display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[0])

#     if len(bboxes) > 0:
#         boxes = np.stack([b[:4] for b in bboxes], axis=0)
#         box_classes = np.array([b[-2] for b in bboxes])
#         mask_indices = np.array([b[-1] for b in bboxes])
#         show_masks = np.stack(masks, axis=-1)[..., mask_indices]
#         class_names = {k: data.coco.cats[k]['name'] for k in data.coco.cats.keys()}
#         display_instances(image, boxes, show_masks, box_classes, class_names, show_bbox=True, ax=ax[1])
#     else:
#         display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[1])


# some trciks to map detectron's dataset_dicts_train to data from copy-paste

# In[22]:


data_id_to_num = {i:q for q,i in enumerate(data.ids)}

ALL_IDS = list(data_id_to_num.keys())

dataset_dicts_train = [i for i in dataset_dicts_train if i['image_id'] in ALL_IDS]

BOX_MODE = dataset_dicts_train[0]['annotations'][0]['bbox_mode']


# In[23]:


import copy
import logging

import detectron2.data.transforms as T
import torch
from detectron2.data import detection_utils as utils

import json
import numpy as np
from pycocotools import mask
from skimage import measure


# Wtite custom mapper

# In[24]:


class MyMapper:
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    def __init__(self, cfg, is_train: bool = True):

        self.is_train = is_train

        mode = "training" if is_train else "inference"
        #print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        img_id = dataset_dict['image_id']
        
        
        aug_sample = data[data_id_to_num[img_id]]
        
        image = aug_sample['image']
        
        image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        
        bboxes = aug_sample['bboxes']
        box_classes = np.array([b[-2] for b in bboxes])
        boxes = np.stack([b[:4] for b in bboxes], axis=0)
        mask_indices = np.array([b[-1] for b in bboxes])
        
        
        masks = aug_sample['masks']
        
        annos = []
        
        for enum,index in enumerate(mask_indices):
            curr_mask = masks[index]
            
            fortran_ground_truth_binary_mask = np.asfortranarray(curr_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(curr_mask, 0.5)
            
            annotation = {
        "segmentation": [],
        "iscrowd": 0,
        "bbox": ground_truth_bounding_box.tolist(), 
        "category_id": train_metadata.thing_dataset_id_to_contiguous_id[box_classes[enum]]  ,
        "bbox_mode":BOX_MODE
                
                
    }
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)
                
            annos.append(annotation)
        

        image_shape = image.shape[:2]  # h, w

        
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


# In[25]:


import os

from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch


# Use custom Trainer

# In[26]:


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=MyMapper(cfg, True), sampler=sampler
        )


# In[28]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine import HookBase
import detectron2.utils.comm as comm
from detectron2.engine import launch

cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("octiva_train",)


cfg.DATASETS.VAL = ("octiva_test",)

batch_size = 16
epochs = 400
no_of_samples = 1454
image_size = 800
no_of_checkpoints_to_keep = 10

max_iter = int(epochs*(no_of_samples/batch_size))
checkpoint_period = int(max_iter*0.05)

                
cfg.INPUT.MIN_SIZE_TEST= 800
cfg.INPUT.MAX_SIZE_TEST = 800
cfg.INPUT.MIN_SIZE_TRAIN = 800
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MASK_FORMAT = "polygon"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0001

cfg.INPUT.FORMAT = 'BGR'
cfg.DATASETS.TEST = ("octiva_test",)
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (checkpoint_period,)
# The iteration number to decrease learning rate by GAMMA.

cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.WARMUP_METHOD = "linear"

cfg.SOLVER.MAX_ITER =max_iter    
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.RETINANET.NUM_CLASSES = 3
cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period


cfg.OUTPUT_DIR = '/home/aboggaram/models/octiva_copypaste_mrcnn_retinanet_101'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = MyTrainer(cfg) 

trainer.resume_or_load(resume=True)


# start train

trainer.train()






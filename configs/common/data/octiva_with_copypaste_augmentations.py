from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
import albumentations as A

# register custom dataset
train_json = '/home/aboggaram/data/Octiva/consolidated_coco_format_validated_11_01_2022/train.json'
test_json = '/home/aboggaram/data/Octiva/consolidated_coco_format_validated_11_01_2022/test.json'
image_dir = '/home/aboggaram/data/Octiva/data_for_playment'
register_coco_instances("octiva_train", {}, train_json, image_dir)
register_coco_instances("octiva_test", {}, test_json, image_dir)



transform = A.Compose(
            aug_list, bbox_params=A.BboxParams(format="coco")
        )


dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="octiva_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(A.OneOf([A.HorizontalFlip(),A.RandomRotate90()],p=0.75)),\
            L(A.OneOf([A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),A.RandomGamma(),A.CLAHE()],p=0.5)),\
            L(A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15)],p=0.5)),\
            L(A.OneOf([A.Blur(),A.MotionBlur(),A.GaussNoise(),A.ImageCompression(quality_lower=75)],p=0.5)),
            L(CopyPaste(blend=True, sigma=1, pct_objects_paste=0.9, p=1.0)), #pct_objects_paste is a guess
            # L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="octiva_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

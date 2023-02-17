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

# register custom dataset
train_json = '/home/aboggaram/data/Octiva/coco_format_with_seg_kpts_Feb_16_2023/train.json'
test_json = '/home/aboggaram/data/Octiva/coco_format_with_seg_kpts_Feb_16_2023/test.json'
image_dir = '/home/aboggaram/data/Octiva/coco_format_with_seg_kpts_Feb_16_2023/images'
register_coco_instances("octiva_train", {}, train_json, image_dir)
register_coco_instances("octiva_test", {}, test_json, image_dir)


dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="octiva_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=640, max_size=1024),
        ],
        image_format="BGR",
        use_instance_mask=True,
        use_keypoint=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="octiva_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=640, max_size=1024),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    kpt_oks_sigmas=[0.33, 0.33, 0.33],
)

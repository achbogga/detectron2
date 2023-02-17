import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

# trunk-ignore(mypy/misc)
from ..common.data.octiva_keypoint import dataloader
# trunk-ignore(mypy/misc)
from ..common.models.keypoint_rcnn_fpn import model
# trunk-ignore(mypy/misc)
from ..common.optim import AdamW as optimizer
# trunk-ignore(mypy/misc)
from ..common.train import train


model_checkpoint_output_dir = "/home/aboggaram/models/Octiva/octiva_keypoint_rcnn_r50_fpn_Feb_15"

num_classes = 1
batch_size = 4
epochs = 650
no_of_train_samples = 4730
no_of_test_samples = 249
image_size = 640
no_of_checkpoints_to_keep = 5
eval_period = 500

max_iter = int(epochs * (no_of_train_samples / batch_size))
checkpoint_period = int(max_iter * 0.05)
eval_steps = int(no_of_test_samples / batch_size)


# train from scratch
train.output_dir = model_checkpoint_output_dir
train.init_checkpoint = \
    '/home/aboggaram/models/Public/kpt_rcnn.pkl'
train.amp.enabled = True
train.ddp.fp16_compression = True
train.checkpointer = dict(period=checkpoint_period,
                          max_to_keep=no_of_checkpoints_to_keep)
train.eval_period = eval_period
model.backbone.bottom_up.freeze_at = 0

# # SyncBN
# # Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# # torch.nn.SyncBatchNorm. We can remove this after
# # https://github.com/pytorch/pytorch/issues/36530 is fixed.
# model.roi_heads.box_head.conv_norm = \
#     model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c,
#                                                                        stats_mode="N")

# model.backbone.bottom_up.stem.norm = \
#     model.backbone.bottom_up.stages.norm = \
#     model.backbone.norm = "SyncBN"

model.roi_heads.num_classes = num_classes

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

# resize_and_crop_image in:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/utils/input_utils.py#L127  # noqa: E501, B950
dataloader.train.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.RandomRotation)(angle=[-30, 30]),
    L(T.RandomSaturation)(intensity_min=0.95, intensity_max=1.05),
    L(T.RandomBrightness)(intensity_min=0.95, intensity_max=1.05),
    L(T.RandomContrast)(intensity_min=0.95, intensity_max=1.05),
]

dataloader.test.mapper.augmentations = [
    L(T.ResizeScale)(
        min_scale=1.0, max_scale=1.0, target_height=image_size, target_width=image_size
    ),
]

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

# larger batch-size.
dataloader.train.total_batch_size = batch_size

# Equivalent to no. of epochs.
train.max_iter = max_iter

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(train.max_iter * 0.6), int(train.max_iter * 0.8)],
        num_updates=train.max_iter,
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

optimizer.lr = 0.0005
optimizer.weight_decay = 4e-5
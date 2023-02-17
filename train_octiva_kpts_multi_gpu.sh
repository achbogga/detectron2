#!/bin/bash
time python3 tools/lazyconfig_train_net.py \
--config-file configs/new_baselines/multi_gpu_iunu_keypoint_rcnn_R_50_FPN_650ep_LSJ.py \
--num-gpus 2 | tee /home/aboggaram/logs/octiva_training_kpts.log
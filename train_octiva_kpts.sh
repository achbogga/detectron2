#!/bin/bash
time python3 tools/lazyconfig_train_net.py --config-file configs/new_baselines/iunu_keypoint_rcnn_R_50_FPN_650ep_LSJ.py --num-gpus 1 | tee /home/aboggaram/logs/octiva_training_kpts.log
#!/bin/bash

case="191015_jma_svglp_normal_train"

python ../plot_comp_prediction.py \
       --model_path ../pretrained_models/191015_jma_svglp_normal_train_model.pth \
       --batch_size 20 --nsample 5 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

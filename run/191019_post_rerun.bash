#!/bin/bash

# ---------------------------------------------------
case="191016_jma_svglp_sqrt_iter120_train.bash"

#python ../plot_comp_prediction.py \
#       --model_path ../logs/$case/model.pth \
#       --batch_size 20 --nsample 5 \
#       --n_past 12 --n_future 12 \
#       --data_root ../datasets/jma/data_kanto_resize/ \
#       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
#       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
#       --log_dir ../logs/$case/plots
#
#python ../post_eval_jma.py \
#       --model_path ../logs/$case/model.pth \
#       --batch_size 20 --nsample 5 \
#       --n_past 12 --n_future 12 \
#       --data_root ../datasets/jma/data_kanto_resize/ \
#       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
#       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
#       --log_dir ../logs/$case
#
## ---------------------------------------------------
case="191016_jma_svglp_sqrt_simple_train.bash"
#
#python ../plot_comp_prediction.py \
#       --model_path ../logs/$case/model.pth \
#       --batch_size 20 --nsample 5 \
#       --n_past 12 --n_future 12 \
#       --data_root ../datasets/jma/data_kanto_resize/ \
#       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
#       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
#       --log_dir ../logs/$case/plots
python ../post_eval_jma.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 --nsample 5 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case



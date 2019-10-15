#!/bin/bash

# ---------------------------------------------------
case="191016_jma_svglp_sqrt_iter120_train.bash"

python ../train_jma_svg_lp.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --z_dim 64 --beta 0.0001 --lr 0.002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 \
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 --nsample 5 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

# ---------------------------------------------------
case="191016_jma_svglp_sqrt_lr002_train.bash"

python ../train_jma_svg_lp.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --z_dim 64 --beta 0.0001 --lr 0.02\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 \
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 --nsample 5 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

# ---------------------------------------------------
case="191016_jma_svglp_sqrt_simple_train.bash"

python ../train_jma_svg_lp.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --z_dim 64 --beta 0.0001 --lr 0.002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 \
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 --nsample 5 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots



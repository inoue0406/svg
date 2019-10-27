#!/bin/bash

# ---------------------------------------------------
case="191023_jma_clstm_sqrt_rnn3"

python ../train_jma_clstm.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --beta 0.0001 --lr 0.0002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 --predictor_rnn_layers 3\
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

python ../post_eval_jma_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/eval


# ---------------------------------------------------
case="191023_jma_clstm_sqrt_lr00002"

python ../train_jma_clstm.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --beta 0.0001 --lr 0.00002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 --predictor_rnn_layers 2\
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

python ../post_eval_jma_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/eval


# ---------------------------------------------------
case="191023_jma_clstm_sqrt_lr0004"

python ../train_jma_clstm.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --beta 0.0001 --lr 0.0004\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 --predictor_rnn_layers 2\
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

python ../post_eval_jma_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/eval


# ---------------------------------------------------
case="191023_jma_clstm_sqrt_iter240"

python ../train_jma_clstm.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --beta 0.0001 --lr 0.0002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 --predictor_rnn_layers 2\
       --epoch_size 100 --niter 240 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

python ../post_eval_jma_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/eval

# ---------------------------------------------------
case="191023_jma_clstm_sqrt_bs30"

python ../train_jma_clstm.py --dataset jma --model dcgan \
       --batch_size 30 --image_width 128 \
       --g_dim 128 --beta 0.0001 --lr 0.0002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 --predictor_rnn_layers 2\
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

python ../post_eval_jma_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/eval

# ---------------------------------------------------
case="191023_jma_clstm_sqrt_flatsampled"

python ../train_jma_clstm.py --dataset jma --model dcgan \
       --batch_size 20 --image_width 128 \
       --g_dim 128 --beta 0.0001 --lr 0.0002\
       --n_past 12 --n_future 12 --n_eval 24 --channels 1 \
       --rnn_size 256 --predictor_rnn_layers 2\
       --epoch_size 100 --niter 120 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case

python ../plot_comp_prediction_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_kanto_flatsampled_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/plots

python ../post_eval_jma_clstm.py \
       --model_path ../logs/$case/model.pth \
       --batch_size 20 \
       --n_past 12 --n_future 12 \
       --data_root ../datasets/jma/data_kanto_resize/ \
       --train_path ../datasets/jma/train_simple_JMARadar.csv \
       --valid_path ../datasets/jma/valid_simple_JMARadar.csv \
       --log_dir ../logs/$case/eval



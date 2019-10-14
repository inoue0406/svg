#!/bin/bash

case="191014_bair_test_run"

python ../train_svg_lp.py --dataset bair --model vgg \
       --batch_size 20 \
       --g_dim 128 --z_dim 64 --beta 0.0001 \
       --n_past 2 --n_future 10 --channels 3 \
       --data_root ../datasets/bair --log_dir ../logs/$case

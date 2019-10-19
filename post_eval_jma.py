#
# Post-evaluation by various criteria trained model
#
import torch
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
import pandas as pd
import h5py
import os
import sys
import random

import itertools
import progressbar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils
from jma_pytorch_dataset import *
from scaler import *
from colormap_JMA import Colormap_JMA

def inv_scaler(x):
    """
    Back to original scale
    """
    return (x ** 2.0)*201.0


def pred_single_model(x,opt,frame_predictor,posterior,prior,encoder,decoder):
    """
    Do one prediction and return as numpy array
    """
    progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    for s in range(opt.nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)

    # prep np.array to be plotted
    GEN = np.zeros([opt.nsample, opt.n_eval, opt.batch_size, 1, opt.image_width, opt.image_width])
    for i in range(opt.n_eval):
        for k in range(opt.nsample):
            GEN[k,i,:,:,:,:] = inv_scaler(all_gen[k][i].cpu().numpy())
    return GEN
    

def eval_batch(x, idx, name,frame_predictor,posterior,prior,encoder,decoder):
    """
    Perform evaluation for single batch
    """
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    # Prep True data
    TRU = np.zeros([opt.n_eval, opt.batch_size, 1, opt.image_width, opt.image_width])
    for i in range(opt.n_eval):
        TRU[i,:,:,:,:] = inv_scaler(x[i].cpu().numpy())
    # perform one prediction
    GEN = pred_single_model(x,opt,frame_predictor,posterior,prior,encoder,decoder)

    import pdb; pdb.set_trace

    
def post_eval_prediction(opt,df_sampled,mode='png_ind'):
    """
    Evaluate the model with several criteria for rainfal nowcasting task

    """
    
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    dtype = torch.cuda.FloatTensor
    
    # ---------------- load the models  ----------------
    tmp = torch.load(opt.model_path)
    frame_predictor = tmp['frame_predictor']
    posterior = tmp['posterior']
    prior = tmp['prior']
    frame_predictor.eval()
    prior.eval()
    posterior.eval()
    encoder = tmp['encoder']
    decoder = tmp['decoder']
    encoder.train()
    decoder.train()
    frame_predictor.batch_size = opt.batch_size
    posterior.batch_size = opt.batch_size
    prior.batch_size = opt.batch_size
    opt.g_dim = tmp['opt'].g_dim
    opt.z_dim = tmp['opt'].z_dim
    opt.num_digits = tmp['opt'].num_digits
    
    # --------- transfer to gpu ------------------------------------
    frame_predictor.cuda()
    posterior.cuda()
    prior.cuda()
    encoder.cuda()
    decoder.cuda()
    
    # ---------------- set the options ----------------
    opt.dataset = tmp['opt'].dataset
    opt.last_frame_skip = tmp['opt'].last_frame_skip
    opt.channels = tmp['opt'].channels
    opt.image_width = tmp['opt'].image_width
    
    print(opt)
    
    # --------- load a dataset ------------------------------------
    # loading datasets
    train_dataset = JMARadarDataset(root_dir=opt.data_root,
                                    csv_file=opt.train_path,
                                    tdim_use=opt.n_past,
                                    transform=None)
    
    valid_dataset = JMARadarDataset(root_dir=opt.data_root,
                                    csv_file=opt.valid_path,
                                    tdim_use=opt.n_past,
                                    transform=None)
    
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    
    test_loader = DataLoader(dataset=valid_dataset,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    
    def get_training_batch():
        while True:
            for sequence in train_loader:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
    training_batch_generator = get_training_batch()
    
    def get_testing_batch():
        while True:
            for sequence in test_loader:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch 
    testing_batch_generator = get_testing_batch()

    for i in range(0, opt.N, opt.batch_size):
        # plot train
        train_x = next(training_batch_generator)
        eval_batch(train_x, i, 'train',frame_predictor,posterior,prior,encoder,decoder)
    
        # plot test
        test_x = next(testing_batch_generator)
        eval_batch(test_x, i, 'test',frame_predictor,posterior,prior,encoder,decoder)
        print(i)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--data_root', default='data', help='root directory for data')
    parser.add_argument('--train_path', default='data', help='csv file containing filenames for training')
    parser.add_argument('--valid_path', default='data', help='csv file containing filenames for validation')
    parser.add_argument('--model_path', default='', help='path to model')
    parser.add_argument('--log_dir', default='', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=28, help='number of frames to predict')
    parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
    parser.add_argument('--nsample', type=int, default=100, help='number of samples')
    parser.add_argument('--N', type=int, default=256, help='number of generated samples')
    parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
    
    opt = parser.parse_args()
    os.makedirs('%s' % opt.log_dir, exist_ok=True)

    opt.n_eval = opt.n_past+opt.n_future
    opt.max_step = opt.n_eval

    # samples to be plotted
    sample_path = '../datasets/jma/sampled_forplot_3day_JMARadar.csv'

    # read sampled data in csv
    df_sampled = pd.read_csv(sample_path)
    print('samples to be plotted')
    print(df_sampled)
    
    post_eval_prediction(opt,df_sampled,mode='png_ind')



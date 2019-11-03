#
# Post-evaluation by various criteria trained model
# for non-probabilistic clstm model
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utils
from jma_pytorch_dataset import *
from scaler import *
from colormap_JMA import Colormap_JMA
from criteria_precip import *

def inv_scaler(x):
    """
    Back to original scale
    """
    return (x ** 2.0)*201.0

def pred_single_model(x,opt,conv_predictor,frame_predictor,encoder,decoder):
    """
    Do one prediction and return as numpy array
    """
    conv_predictor.zero_grad()
    frame_predictor.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()
    
    all_gen = []
    # time step by convlstm
    x_pred_conv = conv_predictor(x)    
    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    x_in = x[0]
    all_gen.append(x_in)
    for i in range(1, opt.n_eval):
#        h = encoder(x_in)
#        if opt.last_frame_skip or i < opt.n_past:	
#            h, skip = h
#        else:
#            h, _ = h
#        h = h.detach()
#        if i < opt.n_past:
#            h_target = encoder(x[i])[0].detach()
#            frame_predictor(h)
#            x_in = x[i]
#            all_gen.append(x_in)
#        else:
#            h = frame_predictor(h.detach())
#            x_in = decoder([h, skip]).detach()
#            all_gen.append(x_in)
        if i < opt.n_past:
            x_in = x[i-1] # use ground truth frame for the first half
            h, skip = encoder(x_in)
            h = h.detach()
        else:
            x_in = x_pred # use predicted frame for the second half (NOT use ground truth)
            _, skip = encoder(x_in)
            h = h_pred
        h_pred = frame_predictor(h).detach()
        x_pred = decoder([h_pred, skip]) + x_pred_conv[:,i,:,:,:]
        x_pred = x_pred.detach()
        all_gen.append(x_pred) 

    # prep np.array to be plotted
    GEN = np.zeros([opt.batch_size, opt.n_eval,  1, opt.image_width, opt.image_width])
    for i in range(opt.n_eval):
        GEN[:,i,:,:,:] = inv_scaler(all_gen[i].cpu().numpy())
    return GEN

def eval_whole_dataset(batch_loader, opt, name, threshold,
                       conv_predictor,frame_predictor,encoder,decoder):
    """
    Perform evaluation for the whole dataset
    """
    # initialize
    emp = np.empty((0,opt.n_eval),float)
    SumSE_all = emp
    hit_all = emp
    miss_all =  emp
    falarm_all = emp
    m_xy_all = emp
    m_xx_all = emp
    m_yy_all = emp
    MaxSE_all =  emp
    FSS_t_all =  emp

    print("total batches to be processed: ",len(batch_loader))
    dtype = torch.cuda.FloatTensor

    ibatch = 0
    for sequence in batch_loader:
        x = utils.normalize_data(opt, dtype, sequence)
        ibatch += 1
        print(name," threshold:",threshold," batch:",ibatch)
        x_in = x[0]
        # Prep True data
        TRU = np.zeros([opt.batch_size, opt.n_eval, 1, opt.image_width, opt.image_width])
        for i in range(opt.n_eval):
            TRU[:,i,:,:,:] = inv_scaler(x[i].cpu().numpy())
        # perform one prediction
        GEN = pred_single_model(x,opt,conv_predictor,frame_predictor,encoder,decoder)
        print(" ground truth max:",np.max(TRU)," gen max:",np.max(GEN))
        
        # Evaluation
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(TRU,GEN,
                                                                  th=threshold)
        FSS_t = FSS_for_tensor(TRU,GEN,th=threshold,win=10)
        SumSE_all = np.append(SumSE_all,SumSE,axis=0)
        hit_all = np.append(hit_all,hit,axis=0)
        miss_all = np.append(miss_all,miss,axis=0)
        falarm_all = np.append(falarm_all,falarm,axis=0)
        m_xy_all = np.append(m_xy_all,m_xy,axis=0)
        m_xx_all = np.append(m_xx_all,m_xx,axis=0)
        m_yy_all = np.append(m_yy_all,m_yy,axis=0)
        MaxSE_all = np.append(MaxSE_all,MaxSE,axis=0)
        FSS_t_all = np.append(FSS_t_all,FSS_t,axis=0)
#        if ibatch==5:
#            break

    # calc metric for the whole dataset
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                                          m_xy_all,m_xx_all,m_yy_all,
                                                          MaxSE_all,FSS_t_all,axis=(0))
    # save evaluated metric as csv file
    tpred = (np.arange(opt.n_eval)+1.0)*5.0 # in minutes
    df = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE,
                       'CSI':CSI,
                       'FAR':FAR,
                       'POD':POD,
                       'Cor':Cor,
                       'MaxMSE': MaxMSE,
                       'FSS_mean': FSS_mean})
    df.to_csv(os.path.join(opt.log_dir,
                           'evaluation_predtime_%s_%.2f.csv' % (name,threshold)))
    
def post_eval_prediction(opt,mode='png_ind'):
    """
    Evaluate the model with several criteria for rainfal nowcasting task

    """
    
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    # ---------------- load the models  ----------------
    tmp = torch.load(opt.model_path)
    conv_predictor = tmp['conv_predictor']
    conv_predictor.eval()
    frame_predictor = tmp['frame_predictor']
    frame_predictor.eval()
    encoder = tmp['encoder']
    decoder = tmp['decoder']
    encoder.train()
    decoder.train()
    conv_predictor.batch_size = opt.batch_size
    frame_predictor.batch_size = opt.batch_size
    opt.g_dim = tmp['opt'].g_dim
    opt.num_digits = tmp['opt'].num_digits
    
    # --------- transfer to gpu ------------------------------------
    conv_predictor.cuda()
    frame_predictor.cuda()
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
    
    for threshold in [0.5,10.0,20.0]:
        #eval_whole_dataset(train_loader, opt, 'train', threshold,
        #                   conv_predictor, frame_predictor, encoder, decoder)
        eval_whole_dataset(test_loader, opt, 'test', threshold,
                           conv_predictor, frame_predictor, encoder, decoder)
        
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
    parser.add_argument('--N', type=int, default=256, help='number of generated samples')
    parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
    
    opt = parser.parse_args()
    os.makedirs('%s' % opt.log_dir, exist_ok=True)

    opt.n_eval = opt.n_past+opt.n_future
    opt.max_step = opt.n_eval
    
    post_eval_prediction(opt,mode='png_ind')



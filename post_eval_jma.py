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

def pred_single_model(x,opt,frame_predictor,posterior,prior,encoder,decoder):
    """
    Do one prediction and return as numpy array
    """
    all_gen = []
    for s in range(opt.nsample):
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
    GEN = np.zeros([opt.nsample, opt.batch_size, opt.n_eval,  1, opt.image_width, opt.image_width])
    for i in range(opt.n_eval):
        for k in range(opt.nsample):
            GEN[k,:,i,:,:,:] = inv_scaler(all_gen[k][i].cpu().numpy())
    return GEN

def eval_whole_dataset(batch_loader, opt, name, threshold,
                       frame_predictor,posterior,prior,encoder,decoder):
    """
    Perform evaluation for the whole dataset
    """
    # initialize
    emp = np.empty((0,opt.n_eval),float)
    SumSE_all = [emp,emp,emp]
    hit_all = [emp,emp,emp]
    miss_all =  [emp,emp,emp]
    falarm_all = [emp,emp,emp]
    m_xy_all = [emp,emp,emp]
    m_xx_all = [emp,emp,emp]
    m_yy_all = [emp,emp,emp]
    MaxSE_all = [emp,emp,emp]
    FSS_t_all = [emp,emp,emp]

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
        GEN = pred_single_model(x,opt,frame_predictor,posterior,prior,encoder,decoder)
        print(" ground truth max:",np.max(TRU)," gen max:",np.max(GEN))
        
        # check best / worst index by MSE
        mse_sample = np.zeros([opt.nsample,opt.n_eval])
        for i in range(opt.batch_size):
            for k in range(opt.nsample):
                mse_sample[k,i] = np.mean((TRU[i] - GEN[k,i]) ** 2)
        id_best = np.argmin(mse_sample,axis=0)
        id_worst = np.argmax(mse_sample,axis=0)
        # get mean, best, and worst prediction
        GEN_mean = np.mean(GEN,axis=0)
        GEN_best = np.zeros(GEN_mean.shape)
        GEN_worst = np.zeros(GEN_mean.shape)
        for i in range(opt.batch_size):
            GEN_best[i] = GEN[id_best[i],i,:,:,:,:]
            GEN_worst[i] = GEN[id_worst[i],i,:,:,:,:]

        # Evaluation
        for l,G in enumerate([GEN_mean,GEN_worst,GEN_best]):
            SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(TRU,G,
                                                                      th=threshold)
            FSS_t = FSS_for_tensor(TRU,G,th=threshold,win=10)
            SumSE_all[l] = np.append(SumSE_all[l],SumSE,axis=0)
            hit_all[l] = np.append(hit_all[l],hit,axis=0)
            miss_all[l] = np.append(miss_all[l],miss,axis=0)
            falarm_all[l] = np.append(falarm_all[l],falarm,axis=0)
            m_xy_all[l] = np.append(m_xy_all[l],m_xy,axis=0)
            m_xx_all[l] = np.append(m_xx_all[l],m_xx,axis=0)
            m_yy_all[l] = np.append(m_yy_all[l],m_yy,axis=0)
            MaxSE_all[l] = np.append(MaxSE_all[l],MaxSE,axis=0)
            FSS_t_all[l] = np.append(FSS_t_all[l],FSS_t,axis=0)
        #if ibatch == 5:
        #    break

    # calc metric for the whole dataset
    for l,txt in enumerate(["0mean","1worst","2best"]):
        RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all[l],hit_all[l],miss_all[l],falarm_all[l],
                                                              m_xy_all[l],m_xx_all[l],m_yy_all[l],
                                                              MaxSE_all[l],FSS_t_all[l],axis=(0))
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
                               'evaluation_predtime_%s_%.2f_%s.csv' % (name,threshold,txt)))
    
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
    
    for threshold in [0.5,10.0,20.0]:
        #eval_whole_dataset(train_loader, opt, 'train', threshold,
        #                   frame_predictor,posterior,prior,encoder,decoder)
        eval_whole_dataset(test_loader, opt, 'test', threshold,
                           frame_predictor,posterior,prior,encoder,decoder)
        
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
    
    post_eval_prediction(opt,mode='png_ind')



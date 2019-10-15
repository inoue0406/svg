#
# Plot Predicted Rainfall Data
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

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)

def plot_rainfall(pic_tg,pic_pred,pic_path,fname,nsample):
    # input
    # pic_tg: numpy array with [time,x,y] dim
    # pic_pred: numpy array with [nsmple,time,x,y] dim
    print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
    # plot
    cm = Colormap_JMA()
    for nt in range(pic_tg.shape[0]):
        fig, ax = plt.subplots(figsize=(50, 8))
        fig.suptitle("Precip prediction starting at: "+fname, fontsize=10)
        #
        id = nt
        dtstr = str((id+1)*5)
        # target
        plt.subplot(1,nsample+1,1)
        im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
        plt.title("true:"+dtstr+"min")
        plt.grid()
        # predicted
        for j in range(nsample):
            plt.subplot(1,nsample+1,j+2)
            im = plt.imshow(pic_pred[j,id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
            plt.title("pred:"+dtstr+"min")
            plt.grid()
        # color bar
        fig.subplots_adjust(right=0.93,top=0.85)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        # save as png
        nt_str = '_dt%02d' % nt
        plt.savefig(pic_path+'/'+'comp_pred_'+fname+nt_str+'.png')
        plt.close()
    

def make_gifs(x, idx, name,frame_predictor,posterior,prior,encoder,decoder):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    for s in range(nsample):
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
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    # prep np.array to be plotted
    TRU = np.zeros([opt.n_eval, opt.batch_size, 1, opt.image_width, opt.image_width])
    GEN = np.zeros([nsample, opt.n_eval, opt.batch_size, 1, opt.image_width, opt.image_width])
    for i in range(opt.n_eval):
        TRU[i,:,:,:,:] = x[i].cpu().numpy()*201.0
        for k in range(nsample):
            GEN[k,i,:,:,:,:] = all_gen[k][i].cpu().numpy()*201.0
    # plot
    for j in range(opt.batch_size):
        plot_rainfall(TRU[:,j,0,:,:],GEN[:,:,j,0,:,:],opt.log_dir,name+"_sample"+str(j),nsample)
    # exit(temp)
    sys.exit()

    progress.finish()
    utils.clear_progressbar()

    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px
        
# plot comparison of predicted vs ground truth
def plot_comp_prediction(opt,df_sampled,mode='png_ind'):
    
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
        make_gifs(train_x, i, 'train',frame_predictor,posterior,prior,encoder,decoder)
    
        # plot test
        test_x = next(testing_batch_generator)
        make_gifs(test_x, i, 'test',frame_predictor,posterior,prior,encoder,decoder)
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
    parser.add_argument('--N', type=int, default=256, help='number of samples')
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
    
    plot_comp_prediction(opt,df_sampled,mode='png_ind')



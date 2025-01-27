# Deterministic forecast
# modified for JMA rainfall data
# Two-layer model combining original svg code and convlstm cell
# 
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np

from jma_pytorch_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--train_path', default='data', help='csv file containing filenames for training')
parser.add_argument('--valid_path', default='data', help='csv file containing filenames for validation')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

# CLSTM EP2 model
import models.convolution_lstm_mod as convlstm_models
hidden_channels=8
kernel_size=3
conv_predictor = convlstm_models.CLSTM_EP2(opt.channels,hidden_channels,kernel_size,
                                           opt.n_past,opt.n_future)

import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

conv_predictor_optimizer = opt.optimizer(conv_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()

# --------- transfer to gpu ------------------------------------
conv_predictor.cuda()
frame_predictor.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

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

#train_loader = DataLoader(train_data,
#                          num_workers=opt.data_threads,
#                          batch_size=opt.batch_size,
#                          shuffle=True,
#                          drop_last=True,
#                          pin_memory=True)
#test_loader = DataLoader(test_data,
#                         num_workers=opt.data_threads,
#                         batch_size=opt.batch_size,
#                         shuffle=True,
#                         drop_last=True,
#                         pin_memory=True)

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

# --------- training funtions ------------------------------------
def train(x):
    conv_predictor.zero_grad()
    frame_predictor.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # time step by convlstm
    x_pred_conv = conv_predictor(x)
    
    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()

    mse = 0
    kld = 0
    for i in range(1, opt.n_past+opt.n_future):
        if i < opt.n_past:
            x_in = x[i-1] # use ground truth frame for the first half
            h, skip = encoder(x_in)
        else:
            x_in = x_pred # use predicted frame for the second half (NOT use ground truth)
            _, skip = encoder(x_in)
            h = h_pred
        h_pred = frame_predictor(h)
        x_pred = decoder([h_pred, skip]) + x_pred_conv[:,i,:,:,:]
        mse += mse_criterion(x_pred, x[i])
        #import pdb; pdb.set_trace()

    loss = mse
    loss.backward()

    conv_predictor_optimizer.step()
    frame_predictor_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future)

# --------- training loop ------------------------------------
# log
flog = open('%s/train_loss.csv' % opt.log_dir,'w')
flog.write('epoch, mse loss\n')

for epoch in range(opt.niter):
    conv_predictor.train()
    frame_predictor.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        mse = train(x)
        epoch_mse += mse

    progress.finish()
    utils.clear_progressbar()

    print('[%02d] mse loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    flog.write('%d, %f\n' % (epoch,epoch_mse/opt.epoch_size))

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'conv_predictor': conv_predictor,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        


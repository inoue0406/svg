# Taken from the following
# https://github.com/automan000/Convolution_LSTM_PyTorch

# Modified to follow encoder-predictor structure

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

class CLSTM_EP2(nn.Module):
    # Conv LSTM with Predictor-only (no encoder) model
    # Predictor's output feeds in as a next input of LSTM cell
    def __init__(self, input_channels, hidden_channels, kernel_size, n_past, n_future):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(CLSTM_EP2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_past = n_past
        self.n_future = n_future
        self.tsize = self.n_past + self.n_future
        self._all_layers = []
        # initialize predictor cell
        cell_p = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.predictor = cell_p
        self._all_layers.append(cell_p)
        # last conv layer for prediction
        self.padding = int((self.kernel_size - 1) / 2)
        self.lastconv = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=True)
        
    def forward(self, input):
        x = input
        bsize, channels, height, width = x[0].size()
        # initialize internal state
        (hp, cp) = self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        # output variable
        xout = Variable(torch.zeros(bsize, self.tsize channels, height, width)).cuda()
        # Predict
        for it in range(self.tsize):
            # forward
            if it < self.n_past:
                x_in = x[it] # use ground truth as last frame
            else:
                x_in = xout[:,it,:,:,:].clone() # input previous timestep's xout
            (hp, cp) = self.predictor(x_in, hp, cp) 
            xout[:,it,:,:,:] = self.lastconv(hp)
        return xout

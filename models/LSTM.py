import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Basic LSTM model like binbinlan previously did , now we only consider single variable output.
    """
    def __init__(self,configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.input = configs.enc_in
        self.pred_len = configs.pred_len
        self.hidden = configs.d_model
        self.out = configs.c_out


        self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden,self.pred_len * self.out)


    def forward(self,x):
        # x: [Batch, Input length, Channel]
        _,(h_n,c_n) = self.lstm(x)
        output = self.fc(h_n[-1])
        output = output.view(x.size(0),self.pred_len,self.out)
        return output

import os
import sys
import time
import argparse
import logging

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from modules import BFBatchNorm1d
from data import dataset
import modules
import util
from data.noise import noise_torch
from data import fr
from data.loss import fnr

#load models
fr_path = '/scratch/bz1030/DS-GA-1013/DeepFreq/checkpoint/model_snr_30_bias_yes/fr/epoch_300.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fr_module, _, _, _, _ = util.load(fr_path, 'fr', device)
fr_module.to(device)
fr_module.eval()

inputs = torch.randn((1, 2, 50)).float()
outputs = fr_module(inputs)
print(outputs.shape)

for idx, layer in enumerate(fr_module.modules()):
    if isinstance(layer, BFBatchNorm1d):
        #print(layer)
        x = torch.randn((2, 64, 100))
        alpha = 2
        y1 = layer(alpha * x)
        y2 = alpha * layer(x)

        print('scale',(y1 - y2).sum().item())

        x = torch.zeros((2, 64,  100))
        y = layer(x)
        y = (y - 0) ** 2
        print('zero', y.sum().item())
        print('BN param', layer.running_mean, layer.running_var,
              layer.weight, layer.bias)

def unit_test(training = True):
    def print_bn_details(bn):
        print(bn.running_mean)
        print(bn.running_var)

    bn_bf = BFBatchNorm1d(5, use_bias=False);
    bn_bias = BFBatchNorm1d(5, use_bias=True);
    print(bn_bf)
    print(bn_bias)

    bn_bf.train()
    bn_bias.train()


    for _ in range(25):
        temp_inp = torch.randn(100, 5, 128) * 10 + 10;
        alpha = 1.1
        bias_out1 = bn_bias(alpha  * temp_inp)
        bias_out2 = alpha * bn_bias(temp_inp)
        bf_out1 = bn_bf(alpha * temp_inp)
        bf_out2 = alpha * bn_bf(temp_inp)
        bias_out = (bias_out1 - bias_out2) ** 2
        bias_out = bias_out.sum().item()
        bf_out = (bf_out1 - bf_out2) ** 2
        bf_out = bf_out.sum().item()
        print('bf', bf_out, 'bias', bias_out)

    print('Bias free', bn_bf.running_mean, bn_bf.running_var,
          bn_bf.weight, bn_bf.bias)

    print('Bias', bn_bias.running_mean, bn_bias.running_var,
          bn_bias.weight, bn_bias.bias)

    bn_bf.eval()
    bn_bias.eval()
    for _ in range(25):
        temp_inp = torch.randn(100, 5, 128) * 10 + 10;
        alpha = 2
        bias_out1 = bn_bias(alpha  * temp_inp)
        bias_out2 = alpha * bn_bias(temp_inp)
        bf_out1 = bn_bf(alpha * temp_inp)
        bf_out2 = alpha * bn_bf(temp_inp)
        bias_out = (bias_out1 - bias_out2) ** 2
        bias_out = bias_out.sum().item()
        bf_out = (bf_out1 - bf_out2) ** 2
        bf_out = bf_out.sum().item()
        print('bf', bf_out, 'bias', bias_out)

#unit_test(True)


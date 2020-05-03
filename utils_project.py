import torch
import numpy as np
from torch.autograd.gradcheck import zero_gradients

def compute_jacobian_and_bias(inputs, net):
    inputs.requires_grad = True
    outputs = net(inputs)

    inp_n1 = inputs.shape[-2]
    inp_n2 = inputs.shape[-1]
    out_n = outputs.shape[-1]

    jacobian = torch.zeros([inp_n1, inp_n2, out_n])

    for i in range(out_n):
        zero_gradients(inputs)
        outputs[0, i].backward(retain_graph=True)
        # print(jacobian[:, :, i].shape,  inputs.grad.data.shape) [2 ,50] === [1, 2, 50]
        jacobian[:, :, i] = inputs.grad.data[0]

    return jacobian.numpy(), inputs.detach().numpy(), outputs.detach().numpy()


def compute_bias(jacobian, inputs, outputs):
    jacobian = jacobian.reshape((100, 1000))
    inputs = inputs.reshape((100))
    pred = jacobian.T.dot(inputs)
    bias = pred.real - outputs
    return bias


def check_bias(fr_module):
    for layer in fr_module.modules():
        if str(layer).startswith('BFBatchNorm1d'):
            x = torch.randn((8, 64, 100))
            alpha = 10
            input1 = alpha * layer(x)
            input2 = layer(x * alpha)
            check = (input1 - input2).sum()
            x = torch.zeros((8, 64, 100))
            input1 = layer(x)
            print(check.item(), input1.sum().item())
import torch.nn as nn
import torch

class BFBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, use_bias=False):
        super(BFBatchNorm1d, self).__init__(num_features, eps, momentum)

        self.use_bias = use_bias;
        if not self.use_bias:
            self.affine = False;

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)

        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** .5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** .5 + self.eps)

        y = self.weight.view(-1, 1) * y
        if self.affine:
            y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


def unit_test():
    def print_bn_details(bn):
        print(bn.running_mean)
        print(bn.running_var)

    bn_bf = BFBatchNorm1d(5, use_bias=False);
    bn_bias = BFBatchNorm1d(5, use_bias=True);

    print('train mode');
    bn_bf.train()
    bn_bias.train()

    for _ in range(25):
        temp_inp = torch.randn(100, 5, 128) * 10 + 10;
        bias_out = bn_bias(temp_inp);
        print('bias: variance %f, mean %f' % (torch.var(bias_out), torch.mean(bias_out)));
        print_bn_details(bn_bias)

        bf_out = bn_bf(temp_inp);
        print('bf: variance %f, mean %f' % (torch.var(bf_out), torch.mean(bf_out)))
        print_bn_details(bn_bf)

    print('eval mode')
    bn_bf.eval()
    bn_bias.eval()

    for _ in range(10):
        temp_inp = torch.randn(100, 5, 128) * 10 + 10;
        bias_out = bn_bias(temp_inp);
        print('bias: variance %f, mean %f' % (torch.var(bias_out), torch.mean(bias_out)));
        print('eval')
        print_bn_details(bn_bias)

        bf_out = bn_bf(temp_inp);
        print('bf: variance %f, mean %f' % (torch.var(bf_out), torch.mean(bf_out)))
        print('eval')
        print_bn_details(bn_bf)



def set_fr_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size,
                    bias=args.bias)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out,
                                            bias=args.bias)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


def set_fc_module(args):
    """
    Create a frequency-counting module
    """
    assert args.fr_size % args.fc_downsampling == 0, \
        'The downsampling factor (fc_downsampling) does not divide the frequency representation size (fr_size)'
    net = None
    if args.fc_module_type == 'regression':
        net = FrequencyCountingModule(n_output=1, n_layers=args.fc_n_layers, n_filters=args.fc_n_filters,
                                      kernel_size=args.fc_kernel_size, fr_size=args.fr_size,
                                      downsampling=args.fc_downsampling, kernel_in=args.fc_kernel_in, bias=args.bias)
    elif args.fc_module_type == 'classification':
        net = FrequencyCountingModule(n_output=args.max_num_freq, n_layers=args.fc_n_layers,
                                      n_filters=args.fc_n_filters, bias=args.bias)
    else:
        NotImplementedError('Counter module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


class PSnet(nn.Module):
    def __init__(self, signal_dim=50, fr_size=1000, n_filters=8, inner_dim=100, n_layers=3, kernel_size=3, bias=False):
        super().__init__()
        self.fr_size = fr_size
        self.num_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim, bias=bias)
        mod = []
        for n in range(n_layers):
            in_filters = n_filters if n > 0 else 1
            mod += [
                nn.Conv1d(in_channels=in_filters, out_channels=n_filters, kernel_size=kernel_size,
                          stride=1, padding=kernel_size // 2, bias=bias),
                nn.BatchNorm1d(n_filters),
                nn.ReLU()
            ]
            print(mod[-1][0])
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(inner_dim * n_filters, fr_size, bias=bias)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, 1, -1)
        x = self.mod(x).view(bsz, -1)
        output = self.out_layer(x)
        return output


class FrequencyRepresentationModule(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25, bias=False):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=bias)
        mod = []
        for n in range(n_layers):
            batchnorm = BFBatchNorm1d
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=bias,
                          padding_mode='circular'),
                batchnorm(n_filters, use_bias=bias),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=bias)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x


class FrequencyCountingModule(nn.Module):
    def __init__(self, n_output, n_layers, n_filters, kernel_size, fr_size, downsampling, kernel_in, bias=False):
        super().__init__()
        mod = [nn.Conv1d(1, n_filters, kernel_in, stride=downsampling, padding=kernel_in - downsampling,
                             padding_mode='circular')]
        for i in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=bias,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        mod += [nn.Conv1d(n_filters, 1, 1)]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(fr_size // downsampling, n_output)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp[:, None]
        x = self.mod(inp)
        x = x.view(bsz, -1)
        y = self.out_layer(x)
        return y


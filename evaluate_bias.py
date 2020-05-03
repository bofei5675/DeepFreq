import os
import argparse
import json
import warnings
import numpy as np
import torch
from data import fr, loss
from data.source_number import aic_arr, sorte_arr, mdl_arr
import util
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.family'] = 'serif'
params = {
    'font.size': 8,
    'legend.fontsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.labelsize': 11,
    'text.usetex': False,
    'text.latex.unicode': False,
    'figure.figsize': [7, 4]
}
matplotlib.rcParams.update(params)
plt.style.use('seaborn-deep')
palette = sns.color_palette('deep', 10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True,
                        help='The data dir. Should contain the .npy files for the tested dB and the frequency file.')
    parser.add_argument('--fr_path1', default=None, type=str, required=True,
                        help='Biased Frequency-representation module path.')
    parser.add_argument('--fr_path2', default=None, type=str, required=True,
                        help='Bias-free Frequency-representation module path.')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='The output directory where the results will be written.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the content of the output directory')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    fr_module1, _, _, _, _ = util.load(args.fr_path1, 'fr', device=device)
    fr_module1.eval()
    fr_module2, _, _, _, _ = util.load(args.fr_path2, 'fr', device=device)
    fr_module2.eval()
    if use_cuda:
        fr_module1.cuda()
        fr_module2.cuda()
    else:
        fr_module1.cpu()
        fr_module2.cpu()

    xgrid = np.linspace(-0.5, 0.5, fr_module1.fr_size, endpoint=False)


    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite:
        raise ValueError('Output directory ({}) already exists and is not empty. Use --overwrite to overcome.'.format(
            args.output_dir))
    elif not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'test.args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model1_fnr_arr, model2_fnr_arr = [], []
    psnet_fnr_arr, psnet_fc_acc = [], []
    model_chamfer, music_aic_chamfer, music_mdl_chamfer = [], [], []
    psnet_chamfer = []
    fc_acc, mdl_acc, aic_acc, sorte_acc = [], [], [], []

    assert os.path.exists(args.data_dir), 'Data directory does not exist'

    with open(os.path.join(args.data_dir, 'data.args'), 'r') as f:
        data_args = json.load(f)
    signal_dim = data_args['signal_dimension']
    num_test = data_args['n_test']

    dB = [float(x) for x in data_args['dB']]
    f = np.load(os.path.join(args.data_dir, 'f.npy'))

    nfreq = np.sum(f >= -0.5, axis=1)

    for k in range(len(dB)):
        data_path = os.path.join(args.data_dir, str(dB[k]) + 'dB.npy')
        if not os.path.exists(data_path):
            warnings.warn('{:.1f}dB data not in data directory.'.format(dB[k]))

        noisy_signals = np.load(data_path)
        noisy_signals = torch.tensor(noisy_signals).to(device)

        with torch.no_grad():
            # Evaluate FNR of the frequency-representation module
            model_fr_torch = fr_module1(noisy_signals)
            model_fr = model_fr_torch.cpu().numpy()
            f_model = fr.find_freq(model_fr, nfreq, xgrid)
            model1_fnr_arr.append(100 * loss.fnr(f_model, f, signal_dim) / num_test)
            # get fr for the second model [bias free]
            model_fr_torch = fr_module2(noisy_signals)
            model_fr = model_fr_torch.cpu().numpy()
            f_model = fr.find_freq(model_fr, nfreq, xgrid)
            model2_fnr_arr.append(100 * loss.fnr(f_model, f, signal_dim) / num_test)

    fig, ax = plt.subplots()
    ax.grid(linestyle='--', linewidth=0.5)
    ax.plot(dB, model1_fnr_arr, label='DF', marker='d', c=palette[3])
    ax.plot(dB, model2_fnr_arr, label='DF_NB', marker='d', c=palette[4])
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('FNR (\%)')
    ax.legend()
    plt.savefig(os.path.join(args.output_dir, 'fnr.png'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

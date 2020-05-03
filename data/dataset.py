import torch
import numpy as np
import torch.utils.data as data_utils
from data import fr, data
from .noise import noise_torch


def load_dataloader(num_samples, signal_dim, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                    kernel_type, kernel_param, batch_size, xgrid, snrl, snrh):
    num_samples *= (snrh - snrl + 1)
    print('Training samples per snr:', num_samples)

    clean_signals, f, nfreq = data.gen_signal(num_samples, signal_dim, max_n_freq, min_sep, distance=distance,
                                              amplitude=amplitude, floor_amplitude=floor_amplitude,
                                              variable_num_freq=True)
    frequency_representation = fr.freq2fr(f, xgrid, kernel_type, kernel_param)

    clean_signals = torch.from_numpy(clean_signals).float()
    f = torch.from_numpy(f).float()
    frequency_representation = torch.from_numpy(frequency_representation).float()
    dataset = data_utils.TensorDataset(clean_signals, frequency_representation, f)
    print('data dimension',  clean_signals.shape, frequency_representation.shape, f.shape)
    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_dataloader_fixed_noise(num_samples, signal_dim, max_n_freq, min_sep, distance, amplitude, floor_amplitude,
                                kernel_type, kernel_param, batch_size, xgrid, snrl, snrh, noise):
    print('Training samples per snr:', num_samples)
    clean_signals_snr = []
    frequency_representation_snr = []
    noisy_signals_snr = []
    f_snr = []
    for i in range(snrl, snrh + 1):
        clean_signals, f, nfreq = data.gen_signal(num_samples, signal_dim, max_n_freq, min_sep, distance=distance,
                                                  amplitude=amplitude, floor_amplitude=floor_amplitude,
                                                  variable_num_freq=True)
        frequency_representation = fr.freq2fr(f, xgrid, kernel_type, kernel_param)

        clean_signals = torch.from_numpy(clean_signals).float()
        f = torch.from_numpy(f).float()
        frequency_representation = torch.from_numpy(frequency_representation).float()
        noisy_signals = noise_torch(clean_signals, i, noise)
        clean_signals_snr.append(clean_signals)
        frequency_representation_snr.append(frequency_representation)
        noisy_signals_snr.append(noisy_signals)
        f_snr.append(f)
    clean_signals_snr = torch.cat(clean_signals_snr, dim=0)
    noisy_signals_snr = torch.cat(noisy_signals_snr, dim=0)
    frequency_representation_snr = torch.cat(frequency_representation_snr, dim=0)
    f_snr = torch.cat(f_snr, dim=0)
    print('data dimension',  clean_signals_snr.shape, noisy_signals_snr.shape, frequency_representation_snr.shape, f_snr.shape)
    dataset = data_utils.TensorDataset(noisy_signals_snr, clean_signals_snr, frequency_representation_snr, f_snr)
    return data_utils.DataLoader(dataset, batch_size=batch_size)


def make_train_data(args):
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    if args.kernel_type == 'triangle':
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param = args.gaussian_std / args.signal_dim
    return load_dataloader(args.n_training, signal_dim=args.signal_dim, max_n_freq=args.max_n_freq,
                           min_sep=args.min_sep, distance=args.distance, amplitude=args.amplitude,
                           floor_amplitude=args.floor_amplitude, kernel_type=args.kernel_type,
                           kernel_param=kernel_param, batch_size=args.batch_size, xgrid=xgrid, snrl=args.snrl, snrh=args.snrh)


def make_eval_data(args):
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    if args.kernel_type == 'triangle':
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param = args.gaussian_std / args.signal_dim
    return load_dataloader_fixed_noise(args.n_validation, signal_dim=args.signal_dim, max_n_freq=args.max_n_freq,
                                       min_sep=args.min_sep, distance=args.distance, amplitude=args.amplitude,
                                       floor_amplitude=args.floor_amplitude, kernel_type=args.kernel_type,
                                       kernel_param=kernel_param, batch_size=args.batch_size, xgrid=xgrid,
                                       snrl=args.snrl, snrh=args.snrh, noise=args.noise)

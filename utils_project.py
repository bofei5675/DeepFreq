import torch
import numpy as np
from torch.autograd.gradcheck import zero_gradients
from data.data import gen_signal
from data.noise import noise_torch
from data import fr
import matplotlib.pyplot as plt
# find nearest
import matplotlib.backends.backend_pdf
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
            
            
def find_neariest_idx(signal_fr, xgrid):
    indices = []
    for i in signal_fr:
        idx = np.argmin(np.abs(i - xgrid))
        indices.append(idx)
    return indices


def svd_jacobian(jacobian):
    fft_filter = jacobian[0]  - 1j * jacobian[1]
    fft_filter = fft_filter.T
    u, s, vh = np.linalg.svd(fft_filter)
    return u, s, vh

def generate_report_plots(num_samples=2, signal_dim=50, min_sep=1., snr=30, fixed_freq=[0.4], save=False):
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    save_dir = './plots'
    num_freq = len(fixed_freq)
    for signal_idx in range(num_samples):
        clean_signals, signal_frs, signal_num_freqs = gen_signal(num_samples=num_samples, 
                                                             signal_dim=signal_dim, 
                                                             num_freq=num_freq, min_sep=min_sep,
                                                            fixed_freq=fixed_freq)
        noisy_signals = noise_torch(torch.as_tensor(clean_signals), snr, 'gaussian')

        clean_signal, signal_fr, signal_num_freq = clean_signals[signal_idx], signal_frs[signal_idx], signal_num_freqs[signal_idx]
        noisy_signal = noisy_signals[signal_idx].cpu().numpy()
        clean_signal_t = clean_signal[0] + clean_signal[1] * 1j 
        clean_signal_fft = np.fft.fft(clean_signal_t, n=1000)
        clean_signal_fft = np.fft.fftshift(clean_signal_fft)
        noisy_signal_t = noisy_signal[0] + 1j * noisy_signal[1]
        noisy_signal_fft = np.fft.fft(noisy_signal_t, n=1000)
        noisy_signal_fft = np.fft.fftshift(noisy_signal_fft)
        if signal_idx == 0: # plot clean first
            noisy_signal = torch.as_tensor(clean_signal).unsqueeze(dim=0)
        else:
            noisy_signal = torch.as_tensor(noisy_signal).unsqueeze(dim=0)

        jacobian, inputs, outputs = compute_jacobian_and_bias(noisy_signal, fr_module)
        fft_filter = jacobian[0] - 1j * jacobian[1]
        
        
        # plot 1
        fig1, ax = plt.subplots(3, 1, figsize=(15, 6))
        xgrid = np.linspace(-0.5, 0.5, fr_module.fr_size, endpoint=False)
        ax[0].plot(xgrid, outputs[0])
        ax[0].set_xticks(np.arange(-0.5, 0.5, 0.2))
        ylim = ax[0].get_ylim()
        for i in range(signal_fr.shape[0]):
            ax[0].vlines(signal_fr[i],ymin=ylim[0], ymax=ylim[1],label='target:{:4.2f}'.format(signal_fr[i]))
        ax[0].legend()
        ax[0].set_xlim(-0.5, 0.5)
        ax[1].plot(xgrid, np.abs(clean_signal_fft),label='clean fft')
        ax[1].plot(xgrid, np.abs(noisy_signal_fft),'--', label='noisy fft')
        ax[1].set_xlim(-0.5, 0.5)
        ylim = ax[1].get_ylim()
        for i in range(signal_fr.shape[0]):
            ax[1].vlines(signal_fr[i],ymin=ylim[0], ymax=ylim[1],label='target:{:4.2f}'.format(signal_fr[i]))
        ax[1].legend()
        im = ax[2].imshow(np.abs(fft_filter))
        ax[2].set_aspect(2.2)
        ax[2].set_ylabel('jacobian', fontsize=13)

        fig1.subplots_adjust(right=0.9)
        cbar_ax = fig1.add_axes([0.91, 0.15, 0.01, 0.7])
        fig1.colorbar(im, cax=cbar_ax)
        plt.show()
        # plot 2
        fig2, ax = plt.subplots(3, 1, figsize=(15, 6))
        xgrid = np.linspace(-0.5, 0.5, fr_module.fr_size, endpoint=False)
        ax[0].plot(xgrid, outputs[0])
        ax[0].set_xticks(np.arange(-0.5, 0.5, 0.2))
        ylim = ax[0].get_ylim()
        for i in range(signal_fr.shape[0]):
            ax[0].vlines(signal_fr[i],ymin=ylim[0], ymax=ylim[1],label='target:{:4.2f}'.format(signal_fr[i]))
        ax[0].legend()

        ax[1].plot(xgrid, np.abs(clean_signal_fft),label='clean fft')
        ax[1].plot(xgrid, np.abs(noisy_signal_fft),'--', label='noisy fft')
        ylim = ax[1].get_ylim()
        for i in range(signal_fr.shape[0]):
            ax[1].vlines(signal_fr[i],ymin=ylim[0], ymax=ylim[1],label='target:{:4.2f}'.format(signal_fr[i]))
        ax[1].legend()
        fft_filter_norm = fft_filter * np.conjugate(fft_filter)
        fft_filter_norm = fft_filter_norm.T.sum(axis=1).real
        ax[2].plot(xgrid, fft_filter_norm)
        ax[0].set_xlim(-0.5, 0.5)
        ax[1].set_xlim(-0.5, 0.5)
        ax[2].set_xlim(-0.5, 0.5)

        fig2.subplots_adjust(right=0.9)
        plt.show()
        
        # plot. 3
        indices = find_neariest_idx(signal_fr, xgrid)
        target_filter = fft_filter.T[indices]
        # time domain
        fig3, ax = plt.subplots(target_filter.shape[0], 2)
        for idx, filt in enumerate(target_filter):
            if target_filter.shape[0] > 1:
                ax[idx, 0].plot(filt.real)
                ax[idx, 1].plot(filt.imag)
                ax[idx, 0].set_title('Real Part of Freq:{:4.2f};'.format(signal_fr[idx]))
                ax[idx, 1].set_title('Complex Part of Freq:{:4.2f};'.format(signal_fr[idx]))
            else:
                ax[0].plot(filt.real)
                ax[1].plot(filt.imag)
                ax[0].set_title('Real Part of Freq:{:4.2f};'.format(signal_fr[idx]))
                ax[1].set_title('Complex Part of Freq:{:4.2f};'.format(signal_fr[idx]))
        plt.tight_layout()
        plt.show()
        
        # fft domian of signals
        fig4, ax = plt.subplots(target_filter.shape[0], 1,  dpi=300)
        for idx, filt in enumerate(target_filter):
            filt_fft = np.fft.fft(filt,n=1000)
            filt_fft = np.fft.fftshift(filt_fft)
            magnitude = np.abs(filt_fft)
            if target_filter.shape[0] > 1:
                ax[idx].plot(xgrid, magnitude)
                ax[idx].plot(signal_fr[idx], magnitude[indices[idx]], '*')
                ax[idx].plot(-signal_fr[idx], magnitude[999-indices[idx]], '*')
                ax[idx].set_title('Freq:{:4.2f}'.format(signal_fr[idx]))
            else:
                ax.plot(xgrid, magnitude)
                ax.plot(signal_fr[idx], magnitude[indices[idx]], '*')
                ax.plot(-signal_fr[idx], magnitude[999-indices[idx]], '*')
                ax.set_title('Freq:{:4.2f}'.format(signal_fr[idx]))
        plt.tight_layout()
        plt.show()
        
        if save:
            file_name = 'output_clean_{}.pdf' if signal_idx == 0 else 'output_{}.pdf'
            pdf = matplotlib.backends.backend_pdf.PdfPages("./plots/" + file_name.format(signal_idx))
            pdf.savefig(fig1)            
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            pdf.savefig(fig4)
            pdf.close()

from scipy.signal import find_peaks
def jac_row_plot(idx, freq_label, filt_non,n=5):
    fig, ax = plt.subplots(dpi=300)

    filt_fft = np.fft.fft(filt_non,n=1000)
    filt_fft = np.fft.fftshift(filt_fft)
    magnitude = np.abs(filt_fft)
    peaks_indices = find_peaks(magnitude)[0]
    peaks_magnitude = magnitude[peaks_indices]
    sorted_indices = np.argsort(peaks_magnitude)[::-1]
    peak_indices = peaks_indices[sorted_indices][:5]
    ax.plot(xgrid, magnitude)
    ax.plot(freq_label, magnitude[idx],  '*', label='freq={:4.4f}'.format(freq_label))
    for sort_idx in peak_indices:
        ax.annotate('{:4.4f}'.format(xgrid[sort_idx]),
                   xy=(xgrid[sort_idx], magnitude[sort_idx]),
                   arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='bottom')
        ax.plot(xgrid[sort_idx], magnitude[sort_idx], '*', color='black')
    ax.set_title('Freq:{:4.4f}'.format(freq_label))
    plt.tight_layout()
    plt.legend()
    return fig, ax

# compute Jacobian for noisy signal
def compute_jacobian_realization(fr_module, num_samples, signal_dim, num_freq, min_sep, fixed_freq, snr):
    clean_signals, signal_frs, signal_num_freqs = gen_signal(num_samples=num_samples, 
                                                         signal_dim=signal_dim, 
                                                         num_freq=num_freq, min_sep=min_sep,
                                                        fixed_freq=fixed_freq)
    noisy_signals = noise_torch(torch.as_tensor(clean_signals), snr, 'gaussian')
    jacobian_realization =  []
    jacobian_realization_clean =  []
    for idx in range(clean_signals.shape[0]):
        clean_signal, signal_fr, signal_num_freq = clean_signals[idx], signal_frs[idx], signal_num_freqs[idx]
        noisy_signal = noisy_signals[idx].cpu().numpy()
        clean_signal_t = clean_signal[0] + clean_signal[1] * 1j 
        clean_signal_fft = np.fft.fft(clean_signal_t, n=1000)
        clean_signal_fft = np.fft.fftshift(clean_signal_fft)
        noisy_signal_t = noisy_signal[0] + 1j * noisy_signal[1]
        noisy_signal_fft = np.fft.fft(noisy_signal_t, n=1000)
        noisy_signal_fft = np.fft.fftshift(noisy_signal_fft)
        noisy_signal = torch.as_tensor(noisy_signal).unsqueeze(dim=0)
        jacobian, inputs, outputs = compute_jacobian_and_bias(noisy_signal, fr_module)
        clean_signal = torch.as_tensor(clean_signal).unsqueeze(dim=0)
        jacobian_clean, inputs, outputs = compute_jacobian_and_bias(clean_signal, fr_module)
        jacobian_realization.append(jacobian)
        jacobian_realization_clean.append(jacobian_clean)
    return jacobian_realization, jacobian_realization_clean

def get_svd(jacobian_realization):
    eigenvals = []
    left_eigenvecs = []
    right_eigenvecs = []
    eff_dims = []
    for jac in jacobian_realization:
        u, s, vh = svd_jacobian(jac)
        eigenvals.append(s)
        left_eigenvecs.append(u)
        right_eigenvecs.append(vh)
        effective_dim = np.sum(s ** 2)
        eff_dims.append(effective_dim)
        print(effective_dim)
    return eigenvals, left_eigenvecs, right_eigenvecs, eff_dims
def plot_eigenvec(n, u, reverse=False):
    fig, ax =  plt.subplots(n, 1, dpi=300)
    for i in range(n):
        if reverse:
            pc1 = u[:, 49-i]
        else:
            pc1 = u[:, i]
        ax[i].plot(xgrid, np.abs(pc1),label='abs', lw=1)
        #ax[i].plot(xgrid, pc1.real, '--*',label='real',markersize=2)
        #ax[i].plot(xgrid, pc1.imag, '--x', label='imag',markersize=1)
    plt.tight_layout()
    return fig
def plot_eigenval(eigenvals):
    fig = plt.figure(dpi=300)
    for each in eigenvals:
        plt.plot(each, color='blue')
    plt.grid()
    return fig

# print PCA results
def pca_plots(jac, fixed_freq, num_samples, signal_dim, idx_comp=0, fixed_freq_idx=0, arbitary_row=None, resolution=1000):
    xgrid = np.linspace(-0.5, 0.5, resolution, endpoint=False)
    if arbitary_row is None:
        target_freq = fixed_freq[fixed_freq_idx]
        #  get target prediction
        row_idx = np.abs(xgrid  - target_freq).argsort()[0]
    else:
        arbitary_row = int((arbitary_row / 1000) * resolution)
        row_idx = arbitary_row
        target_freq = xgrid[arbitary_row]
    dataset = 1j * np.zeros((num_samples, signal_dim))
    for i in range(num_samples):
        jacobian = jac[i]
        dataset[i] = jacobian[0, :, row_idx] - 1j * jacobian[1, : ,row_idx]
    signal_cov = np.cov(dataset.T)
    s, u = np.linalg.eig(signal_cov)
    pc1 = u[:, idx_comp]
    pc1_fft = np.fft.fft(pc1, n=resolution)
    pc1_fft = np.fft.fftshift(pc1_fft)
    pc1_fft = np.abs(pc1_fft)
    fig, ax = plt.subplots()
    ax.plot(xgrid, pc1_fft)
    ax.set_title('Freq {:2.4f}; PC:{}'.format(target_freq, idx_comp))
    return fig, ax, s, u


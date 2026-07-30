#!usr/bin/python

# Based on description of canonical algo in:
#   Entropy Based Voice Activity Detection in Very Noisy Conditions by Phillipe Renevey

import numpy as np
import matplotlib.pyplot as plt

#from scipy.io import wavfile 
import soundfile as sf
import torch


def is_speech_frame(short_time_energy, energy_threshold_noise,
                    energy_threshold_speech, prev_vad_decision):

    # for hysteresis when between thresholds
    vad_decision = prev_vad_decision

    # if noise or speech threshold crossed change vad decision
    if short_time_energy > energy_threshold_speech:
        vad_decision = 1
    if short_time_energy < energy_threshold_noise:
        vad_decision = 0

    return vad_decision


def estimate_noise_energy_noise_seg(short_time_energy_frame,
                                    noise_energy_prev_frame,
                                    scale_factor):

    noise_energy = scale_factor * noise_energy_prev_frame         \
                   + (1 - scale_factor) * short_time_energy_frame

    return noise_energy


def estimate_noise_energy_speech_seg(short_time_energy_frame,
                                     noise_energy_prev_frame,
                                     scale_factor):

    noise_energy = scale_factor * noise_energy_prev_frame         \
                    + (1 - scale_factor) * short_time_energy_frame

    return noise_energy


def update_noise_threshold(
        delta_n, noise_energy_noise_seg):

    T_n = noise_energy_noise_seg + delta_n

    return T_n


def update_speech_threshold(
        delta_s, noise_energy_speech_seg):

    T_s = noise_energy_speech_seg + delta_s

    return T_s

def run_energy_vad(frames, ad_sf_n, ad_sf_s, d_n, d_s, plot_output = False):
    # Accept both torch tensors and numpy arrays
    is_torch = torch.is_tensor(frames)
    if is_torch:
        device = frames.device
        dtype = frames.dtype
        frames_np = frames.cpu().numpy()
    else:
        device = None
        dtype = None
        frames_np = frames
    
    nframes = frames_np.shape[1]
    # calculate_short_time_energy(frame)
    # square and sum values in column
    e_st = np.sum(frames_np**2, axis=0)
    T_n = np.zeros(nframes)
    T_s = np.zeros(nframes)
    e_noise = np.zeros(nframes)

    # initialise threshold for noise/speech identification
    T_n[0] = e_st[0]
    T_s[0] = e_st[0]

# initialise vad_decision
    vad_decision = np.zeros(nframes)

# iterate frames
    for n in range(0, nframes-1):

        # assume first frame to be noise
        if n == 0:
            e_noise[n] = e_st[0]
        elif vad_decision[n-1] == 0:
            e_noise[n] = estimate_noise_energy_noise_seg(
                e_st[n], e_noise[n-1], ad_sf_n)
        elif vad_decision[n-1] == 1:
            e_noise[n] = estimate_noise_energy_speech_seg(
                e_st[n], e_noise[n-1], ad_sf_s)

        T_n[n] = update_noise_threshold(d_n, e_noise[n])
        T_s[n] = update_speech_threshold(d_s, e_noise[n])

        vad_decision[n] = is_speech_frame(
            e_st[n], T_n[n], T_s[n], vad_decision[n-1])

        print_diagnostics = False
        if print_diagnostics is True:
            print("n: ", n)
            print("T_n: ", T_n[n])
            print("T_s: ", T_s[n])
            print("e_st: ", e_st[n])
            print("e_noise: ", e_noise[n])
            print("vad_decision: ", vad_decision[n])
    
    # Convert back to torch if input was torch
    if is_torch:
        e_st = torch.from_numpy(e_st).to(device=device, dtype=dtype)
        e_noise = torch.from_numpy(e_noise).to(device=device, dtype=dtype)
        T_n = torch.from_numpy(T_n).to(device=device, dtype=dtype)
        T_s = torch.from_numpy(T_s).to(device=device, dtype=dtype)
        vad_decision = torch.from_numpy(vad_decision).to(device=device)

    if plot_output is True:
        def _np(t):
            return t.detach().cpu().numpy() if torch.is_tensor(t) else t

        e_st_np = _np(e_st)
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        ax.plot(e_st_np, label='energy')
        ax.plot(_np(e_noise), label='noise estimate')
        ax.plot(_np(T_n), label='noise threshold')
        ax.plot(_np(T_s), label='speech threshold')
        ax.plot(_np(vad_decision) * np.max(e_st_np), label='energy vad (scaled)')
        ax.set_title('Adaptive Energy VAD (Paper Sec. 3)')
        ax.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig('vad_output.png', dpi=300)
        plt.show()
    return e_st, e_noise, T_n, T_s, vad_decision



# read input data from input file

# 16 kHz
# data, fs = sf.read('C:/repo/vad/vad_nrg/test_vectors/sample.wav')
# data, fs = sf.read(
#    'C:/audio/NORMALIZE_ONLY_PRUNED_16k/AlanYeung/ISO_KEY_006_in_2014-03-03--13-37-44.wav'
#)
# data, fs = sf.read('C:/audio/NORMALIZE_ONLY_PRUNED_16k/AliErtugul/ISO_KEY_006_in_2014-03-03--09-48-12.wav')
# data, fs = sf.read('C:/audio/NORMALIZE_ONLY_PRUNED_16k/ConorGardiner/ISO_KEY_006_in_2014-02-28--11-36-50.wav')

# fname = "C:/repo/vad/vad_nrg/test_vectors/NOIZEUS/files/clean/sp01.wav"
# fname = "C:/repo/vad/vad_nrg/test_vectors/NOIZEUS/files/restaurant_5dB/5dB/sp19_restaurant_sn5.wav"
#fname = "C:/repo/vad/vad_nrg/test_vectors/NOIZEUS/files/babble_0dB/0dB/sp01_babble_sn0.wav"
fname = "/simulated/scratch/projects/interns-work/members/athorne/generated_datasets/16kHz/DNS3/run2/valid/valid_noisy/000002_001.wav"

# fname = 'C:/audio/NORMALIZE_ONLY_PRUNED_16k/ConorGardiner/ISO_KEY_006_in_2014-02-28--11-36-50.wav'
# fname = 'C:/audio/NORMALIZE_ONLY_PRUNED_16k/richardmunro/ISO_KEY_006_in_2014-03-06--15-27-29.wav'
# fname = 'C:/audio/NORMALIZE_ONLY_PRUNED_16k/AlanYeung/ISO_KEY_006_in_2014-03-03--13-37-44.wav'

print(sf.check_format(fname))
print(sf.info(fname))

data, fs = sf.read(fname)

# Convert stereo to mono if needed
data = data[:, 0] if data.ndim == 2 else data

print(sf.check_format(fname))
print(sf.info(fname))

do_plot_input = True
if do_plot_input is True:
    plt.figure()
    plt.plot(data)
    plt.title('input audio')
    plt.show()

# define symbols

# N        : No of samples per frame (Should be 10 ms frames)
# ad_sf_n  : adaptive scale factor for noise segment range = [0.85,0.95]
# ad_sf_s  : adaptive scale factor for speech segment range = [0.98, 0.999]
# e_st     : short term energy per frame
# e_noise  : estimate of noise energy
# T_n      : noise threshold
# T_s      : speech threshold
# d_n      : additive constant for noise threshold calc range [0.1, 0.4]
# d_s      : additive constant for speech threshold calc range [0.5,0.8]

# default tuning (16kHz)
N = 160
ad_sf_n = 0.98
ad_sf_s = 0.999
d_n = 0.4
d_s = 0.5

# default tuning (8kHz)
##N = 80
##ad_sf_n = 0.98
##ad_sf_s = 0.999
##d_n = 0.4
##d_s = 0.5

old_size = data.size
new_size = old_size + N - old_size % N
x0 = np.zeros(new_size)
x0[:data.size] = data

# sort input data into frames (column per frame)
x = x0.reshape(N, -1)
x = x.T

nframes = x.shape[1]

# calculate_short_time_energy(frame)
# square and sum values in column
e_st = np.sum(x**2, axis=0)

# pdb.set_trace()

e_st, e_noise, T_n, T_s, vad_decision = run_energy_vad(
    x, ad_sf_n, ad_sf_s, d_n, d_s)

plot_output = True
if plot_output is True:
    fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    axs[0].plot(data)
    axs[0].set_title('Input Audio')

    axs[1].plot(e_st, label='energy')
    axs[1].plot(e_noise, label='noise estimate')
    axs[1].plot(T_n, label='noise threshold')
    axs[1].plot(T_s, label='speech threshold')
    axs[1].plot(vad_decision * np.max(e_st), label='energy vad (scaled)')
    axs[1].set_title('Adaptive Energy VAD (Paper Sec. 3)')
    axs[1].legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('vad_output.png', dpi=300)
    plt.show()
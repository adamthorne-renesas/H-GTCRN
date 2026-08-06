import numpy as np
import matplotlib.pyplot as plt

#from scipy.io import wavfile 
import soundfile as sf
import torch

#https://www.sciencedirect.com/science/article/pii/S1051200423002464#se0050

N_FFT = 512
HOP_LEN_A = 160
HOP_LEN_B = 320
WIN_LEN_A = 320
WIN_LEN_B = 640
input_dir = "test/in"
output_dir = "test/out_lw"

def preemphasis(x, coeff=0.97):
   out = x.clone()
   out[:,1:] = x[:, 1:] - coeff * x[:, :-1]
   return out

def frame_signal(x, frame_length, hop_length):
    C, L = x.shape
    num_frames = 1 + (L - frame_length) // hop_length
    frames = torch.zeros((C, num_frames, frame_length), dtype=x.dtype, device=x.device)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[:, i, :] = x[:, start:end]
    return frames
    
def apply_window(frames):
    frame_length = frames.shape[-1]
    hamming = torch.hamming_window(frame_length, dtype=frames.dtype, device=frames.device)
    return frames * hamming

def compute_stft(windowed_frames, n_fft):
    stft = torch.fft.rfft(windowed_frames, n=n_fft, dim=-1)
    return stft

def hz_to_mel(hz):
    hz = torch.tensor(hz, dtype=torch.float32) if not isinstance(hz, torch.Tensor) else hz
    return 2595 * torch.log10(1 + hz / 700)

def mel_to_hz(mel):
    mel = torch.tensor(mel, dtype=torch.float32) if not isinstance(mel, torch.Tensor) else mel
    return 700 * (10 ** (mel / 2595) - 1)

def mel_filter_bank(n_mels, n_fft, sample_rate, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2
    n_freqs = n_fft // 2 + 1
    fmin_mel = hz_to_mel(fmin)
    fmax_mel = hz_to_mel(fmax)
    mel_points = torch.linspace(fmin_mel, fmax_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    fb=torch.zeros((n_mels, n_freqs), dtype=torch.float32)

    for m in range(1, n_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]
            if center == left:
                center += 1
            if right == center:
                right += 1
            for k in range(left, center):
                fb[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                fb[m - 1, k] = (right - k) / (right - center)
    return fb

def dct_matrix(n_mels, device, dtype):
    # DCT-II orthonormal matrix, shape (n_mels, n_mels)
    n = torch.arange(n_mels, dtype=dtype, device=device)
    k = n.unsqueeze(1)
    M = torch.cos(torch.pi / n_mels * (n + 0.5) * k)
    M *= torch.sqrt(torch.tensor(2.0 / n_mels))
    M[0] *= 1 / torch.sqrt(torch.tensor(2.0))   # ortho normalization for k=0
    return M   # apply via  log_mel @ M.T

def mfcc(x, n_mels=26, n_mfcc_start=2, n_mfcc_end=14, N_FFT=512):
    power = torch.abs(x) ** 2
    mel_fb = mel_filter_bank(n_mels=n_mels, n_fft=N_FFT, sample_rate=16000)
    mel_energy = torch.matmul(power, mel_fb.T)
    mfcc_log = torch.log(mel_energy + 1e-10)
    M = dct_matrix(n_mels, device=mel_energy.device, dtype=mel_energy.dtype)
    mfcc = torch.matmul(mfcc_log, M.T)
    return mfcc[..., n_mfcc_start:n_mfcc_end]

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def LPC(x, order=12):
    C, T, Lf = x.shape
    lpc_coeffs = torch.zeros((C, T, order + 1), dtype=x.dtype, device=x.device)
    for c in range(C):
        for t in range(T):
            frame = x[c, t, :].cpu().numpy()
            R = np.correlate(frame, frame, mode='full')[len(frame)-1:]
            R = R[:order + 1]
            a = np.zeros(order + 1)
            E = R[0]
            a[0] = 1.0
            for i in range(1, order + 1):
                k = (R[i] - np.dot(a[1:i], R[i-1:0:-1])) / E
                a[1:i] -= k * a[i-1:0:-1]
                a[i] = k
                E *= (1 - k ** 2)
            lpc_coeffs[c, t, 1:] = torch.from_numpy(a[1:]).to(x.device, dtype=x.dtype)
    return lpc_coeffs

def NSCC(x, sample_rate, fmax=None, N_FFT=512, band_low=0, band_high=300):
    power = torch.abs(x) ** 2
    C, T, F = power.shape
    if band_high > F:
        band_high = F
    if fmax is None:
        fmax = sample_rate / 2    
    band_power  = power[:, :, band_low:band_high]
    f = torch.arange(band_low, band_high, device=x.device, dtype=power.dtype)
    b = f * sample_rate / (N_FFT)
    numerator = torch.sum(band_power * b, dim=-1)
    denominator = torch.sum(band_power, dim=-1)
    scc = numerator / (denominator + 1e-10)   
    l_m = band_low * sample_rate / (N_FFT)
    h_m = (band_high - 1) * sample_rate / (N_FFT)
    nscc = scc - (h_m + l_m) / (2.0 * (h_m - l_m + 1e-10))
    return nscc.unsqueeze(-1)

def lengthpad(x, target):
    C, T, D = x.shape
    _, NT, _ = target.shape
    position = torch.arange(NT, device=x.device)
    index = torch.round(position * (T - 1) / (NT - 1)).long()
    clamped = torch.clamp(index, 0, T - 1)
    x = x[:, clamped, :]
    return x

def fuse(x, sample_rate):
    preemph = preemphasis(x)
    apply_windowed_frames_A = apply_window(frame_signal(preemph, WIN_LEN_A, HOP_LEN_A))
    apply_windowed_frames_B = apply_window(frame_signal(preemph, WIN_LEN_B, HOP_LEN_B))
    compute_stft_A = compute_stft(apply_windowed_frames_A, N_FFT)
    compute_stft_B = compute_stft(apply_windowed_frames_B, N_FFT)
    mfcc_A = mfcc(compute_stft_A)
    mfcc_B = mfcc(compute_stft_B)
    lengthpad_mfcc_B = lengthpad(mfcc_B, mfcc_A)
    lpc_a = LPC(apply_windowed_frames_A)[:, :, 1:]
    nscc_a = NSCC(compute_stft_A, sample_rate, N_FFT=N_FFT, band_low=0, band_high=300)
    fused = torch.cat([mfcc_A, lengthpad_mfcc_B, lpc_a, nscc_a], dim=-1)
    return fused


C = 2
sample_rate = 16000
duration = 1
L = sample_rate * duration

x =torch.randn(C, L)  

fused = fuse(x, sample_rate)
print(fused.shape)  # Should print (C, T, D) where T is the number of frames and D is the feature dimension
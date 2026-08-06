#RESEARCH
#Open Access
#Efficient voice activity detection algorithm
#using long-term spectral flatness measure

import argparse
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

N_FFT = 512
HOP_LEN = 256
WIN_LEN = 512
input_dir = "test/in"
output_dir = "test/out_sf_br"

##python vad_sf_br.py --alpha 0.45 --beta 0.55 --ema 0.9 --t_low 0.4 --t_high 0.7 --band_start 61 --band_end 108

def projection_back(Y, ref):
    num = torch.sum(torch.conj(ref[:, :, :, None]) * Y, dim=1)
    denom = torch.sum(torch.abs(Y) ** 2, dim=1)

    c = torch.ones_like(num)
    I = denom > 0.0
    c[I] = num[I] / denom[I]

    return c

def auxiva(X, n_src=None, n_iter=20, proj_back=True, W0=None, model="laplace"):
    """Auxiliary-function-based Independent Vector Analysis (AuxIVA)"""
    n_batches, n_frames, n_freq, n_chan = X.shape
    device = X.device
    if n_src is None:
        n_src = n_chan

    W_hat = torch.zeros((n_batches, n_freq, n_chan, n_chan), dtype=X.dtype).to(device)
    W = W_hat[:, :, :n_src, :]

    if W0 is None:
        W[:, :, :, :n_src] = torch.tile(
            torch.eye(n_src, n_src, dtype=X.dtype), (n_batches, n_freq, 1, 1)
        ).to(device)
    else:
        W[:, :, :, :] = W0

    eps = 1e-10
    eyes = torch.tile(
        torch.eye(n_chan, n_chan, dtype=X.dtype), (n_batches, n_freq, 1, 1)
    ).to(device)

    r_inv = torch.zeros((n_batches, n_src, n_frames)).to(device)
    r = torch.zeros((n_batches, n_src, n_frames)).to(device)

    Y = torch.zeros((n_batches, n_freq, n_src, n_frames), dtype=X.dtype).to(device)

    X_original = X
    X = X.permute(0, 2, 3, 1).clone()

    def demix(Y, X, W):
        Y[:, :, :, :] = torch.matmul(W, X)

    for _ in range(n_iter):
        demix(Y, X, W)

        if model == "laplace":
            r[:, :, :] = 2.0 * torch.norm(Y, dim=1)
        elif model == "gauss":
            r[:, :, :] = (torch.norm(Y, dim=1) ** 2) / n_freq

        r[r < eps] = eps
        r_inv[:, :, :] = 1.0 / r

        for s in range(n_src):
            V = (
                torch.matmul(
                    (X * r_inv[:, None, s, None, :]), torch.conj(X.swapaxes(2, 3))
                )
                / n_frames
            )

            WV = torch.matmul(W_hat, V)
            W[:, :, s, :] = torch.conj(
                torch.linalg.solve(WV + eps * eyes, eyes[:, :, :, s])
            )

            denom = torch.matmul(
                torch.matmul(W[:, :, None, s, :], V[:, :, :, :]),
                torch.conj(W[:, :, s, :, None]),
            )
            # Paper-faithful IVA weight normalization: divide by the complex sqrt of
            # denom (Wang et al.). Kept out-of-place (assignment instead of `/=`) to
            # avoid the fragile in-place complex divide CUDA path on some cu118 builds.
            W[:, :, s, :] = W[:, :, s, :] / torch.sqrt(
                denom[:, :, :, 0] + eps * torch.ones((n_batches, n_freq, 1), device=device)
            )

    demix(Y, X, W)

    Y = Y.permute(0, 3, 1, 2).clone()

    if proj_back:
        z = projection_back(Y, X_original[:, :, :, 0])
        Y *= torch.conj(z[:, None, :, :])
    
    return Y

class vad_sf_br:
    def __init__(self, window=11, ltsf2_window=11):
        self.window = window
        self.ltsf2_window = ltsf2_window

    def _moving_flatness(self, x, window):
        # x is (B, C, T); compute causal moving flatness along time.
        eps = 1e-10
        B, C, T = x.shape
        w = max(1, min(int(window), T))
        if w == 1:
            return x

        x_bc = x.reshape(B * C, 1, T)
        padded = torch.nn.functional.pad(x_bc, (w - 1, 0), mode='replicate')
        frames = padded.unfold(-1, w, 1)  # (B*C, 1, T, w)
        geom = torch.exp(torch.mean(torch.log(frames + eps), dim=-1))
        arith = torch.mean(frames, dim=-1)
        out = geom / (arith + eps)

        return out.reshape(B, C, T)

    def compute_ltsf(self, signal):
        # Compute the magnitude spectrum
        magnitude = torch.abs(signal).pow(2)
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-10), dim=-2))
        arithmetic_mean = torch.mean(magnitude, dim=-2)
        ltsf = geometric_mean / (arithmetic_mean + 1e-10)

        # Second-order LTSF: temporal flatness of the first LTSF track.
        ltsf2 = self._moving_flatness(ltsf, self.ltsf2_window)
        return ltsf2
    
    def compute_band_ratio(self,signal, band_start, band_end):
        # Compute the magnitude spectrum
        magnitude = torch.abs(signal).pow(2)
        
        # Compute the band ratio
        band_energy = torch.sum(magnitude[:, :, band_start:band_end, :], dim=-2)
        total_energy = torch.sum(magnitude, dim=-2)
        band_ratio = band_energy / (total_energy + 1e-10)
        
        return band_ratio
    
    def detect_voice_activity(self, signal, alpha, beta, band_start=9, band_end=108,
                              ema=0.9, t_low=0.4, t_high=0.6, switch_margin=0.05):

        band_ratio = self.compute_band_ratio(signal, band_start, band_end)

        
        ltsf = self.compute_ltsf(signal)   # (B, C, T), unsmoothed
        B, C, T = ltsf.shape
        kernel = torch.ones(1, 1, self.window, device=ltsf.device, dtype=ltsf.dtype) / self.window
        padded = torch.nn.functional.pad(
        ltsf.reshape(B * C, 1, T), (self.window - 1, 0), mode='replicate')  # left-pad only
        ltsf = torch.nn.functional.conv1d(padded, kernel).reshape(B, C, T)

        # Per-frame speech-likeness score for every channel.
        decision = alpha * (1 - ltsf) + beta * band_ratio             # (B, C, T)

        # Streaming: decide every frame using only current + past data.
        B, C, T = decision.shape
        device = decision.device
        idx = torch.arange(B, device=device)
        hist = torch.zeros_like(decision)

        score = torch.zeros(B, C, device=device)                      
        selected = torch.zeros(B, T, dtype=torch.long, device=device)
        vad = torch.zeros(B, T, device=device)
        prev_state = torch.zeros(B, C, dtype=torch.long, device=device)
        cur_channel = torch.zeros(B, dtype=torch.long, device=device) 

        for t in torch.arange(T):
            score = ema * score + (1 - ema) * decision[:, :, t]

            # Per-channel VAD state from the same score used for channel selection.
            state = prev_state.clone()
            state[score > t_high] = 1
            state[score < t_low] = 0

            active = state.bool()
            masked_score = torch.where(active, score, torch.full_like(score, -1e9))
            best_active = masked_score.argmax(dim=1)
            best_score = score.argmax(dim=1)
            has_active = active.any(dim=1)
            candidate = torch.where(has_active, best_active, best_score)

            if t == 0:
                cur_channel = candidate
            else:
                switch = score[idx, candidate] > score[idx, cur_channel] + switch_margin
                cur_channel = torch.where(switch, candidate, cur_channel)

            selected[:, t] = cur_channel
            vad[:, t] = state[idx, cur_channel].to(vad.dtype)
            prev_state = state
            hist[:, :, t] = score

        return ltsf, band_ratio, hist, selected, vad, decision, prev_state

def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t

def main():
    parser = argparse.ArgumentParser(description="Plot vad_sf_br outputs for a wav file")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--ema", type=float, default=0.9)
    parser.add_argument("--t_low", type=float, default=0.4)
    parser.add_argument("--t_high", type=float, default=0.6)
    parser.add_argument("--band_start", type=int, default=9)
    parser.add_argument("--band_end", type=int, default=108)
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--ltsf2_window", type=int, default=11)
    args = parser.parse_args()
    device = torch.device("cpu")
    
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(".wav")
        ]
    )

    if not wav_files:
        raise FileNotFoundError(f"No wav files found in input_dir: {input_dir}")
    for wav_name in wav_files:
        in_path = os.path.join(input_dir, wav_name)
        base_name, ext = os.path.splitext(wav_name)
        out_name = f"{base_name}{ext}"
        out_path = os.path.join(output_dir, out_name)

        try:
            signal, fs = sf.read(in_path)
            wav = torch.from_numpy(signal.T).float()   # (C, L) or (L,)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)                 # (1, L)

            
            window = torch.hann_window(WIN_LEN)
            spec = torch.stft(
                    wav, N_FFT, HOP_LEN, WIN_LEN, window, onesided=True, return_complex=True
                )                                      # (C, F, T)
            iva_in = spec.permute(2, 1, 0).unsqueeze(0)  # (B=1, T, F, C)
            iva_out = auxiva(iva_in).to(device)          # (B=1, T, F, C)
            spec_vad = iva_out.permute(0, 3, 2, 1)       # (B=1, C, F, T)
            ltsf, band_ratio, score, selected_channel, vad, decision, prev = vad_sf_br(
                window=args.window, ltsf2_window=args.ltsf2_window
            ).detect_voice_activity(
                spec_vad, alpha=args.alpha, beta=args.beta, ema=args.ema,
                t_low=args.t_low, t_high=args.t_high,
                band_start=args.band_start, band_end=args.band_end,
            )

            # Reconstruct separated channels from AuxIVA output for final channel selection.
            sep_ch = []
            for c in range(spec_vad.shape[1]):
                ch_spec = iva_out[0, :, :, c].transpose(0, 1).contiguous()  # (F, T)
                sep_ch.append(torch.istft(ch_spec, N_FFT, HOP_LEN, WIN_LEN, window, length=wav.shape[1]))
            sep_wav = torch.stack(sep_ch, dim=0)                           # (C, L)

            # Output audio = the per-frame selected channel only.
            sel = selected_channel.squeeze(0).cpu().numpy()         # (T,)
            wav_np = sep_wav.cpu().numpy()                          # (C, L)
            L = wav_np.shape[1]
            sel_samples = np.repeat(sel, HOP_LEN)
            if sel_samples.shape[0] < L:
                sel_samples = np.pad(sel_samples, (0, L - sel_samples.shape[0]), mode='edge')
            else:
                sel_samples = sel_samples[:L]
            out_audio = wav_np[sel_samples, np.arange(L)]
            sf.write(out_path, out_audio, fs)
            print(f"[OK] {in_path} -> {out_path}")

            plot_output = True  # Set to True to enable plotting of VAD outputs
            if plot_output:
                C = ltsf.shape[1]
                fig, axs = plt.subplots(4, 1, figsize=(11, 20), sharex=False)
                fig.suptitle(f"VAD Outputs for a={args.alpha}_b={args.beta}_tl={args.t_low}_th={args.t_high}_bs={args.band_start}_be={args.band_end}")
                for ch in range(wav_np.shape[0]):
                    axs[0].plot(wav_np[ch], label=f'ch{ch + 1}')
                axs[0].set_title('Input Audio (noisy)')
                axs[0].legend(loc='upper right')

                for i in range(2):
                    ax = axs[1 + i]
                    if i < C:
                        ax.plot(_np(ltsf)[0, i], label='LTSF')
                        ax.plot(_np(band_ratio)[0, i], label='Band Ratio')
                        ax.plot(_np(score)[0, i], label='Score')
                    ax.set_title(f'Channel {i + 1}')
                    ax.legend(loc='upper right')

                axs[3].plot(_np(selected_channel)[0], label='Selected channel')
                axs[3].set_title('VAD Decision')
                axs[3].legend(loc='upper right')

                fig.tight_layout()
                output_subdir = os.path.join(output_dir, f'a={args.alpha}_b={args.beta}_ema={args.ema}_tl={args.t_low}_th={args.t_high}_bs={args.band_start}_be={args.band_end}')
                os.makedirs(output_subdir, exist_ok=True)
                plt.savefig(os.path.join(output_subdir, f'{base_name}_vad_sf_br.png'), dpi=300)
                plt.close(fig)
        except Exception as e:
             print(f"[FAILED] {in_path}: {e}")
if __name__ == "__main__":
    main()
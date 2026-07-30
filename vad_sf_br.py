#RESEARCH
#Open Access
#Efficient voice activity detection algorithm
#using long-term spectral flatness measure

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

class vad_sf_br:
    def __init__(self, window=11):
        self.window = window

    def compute_ltsf(self, signal):
        # Compute the magnitude spectrum
        magnitude = torch.abs(signal).pow(2)
        
        # Compute the long-term spectral flatness measure (LTSF)
        geometric_mean = torch.exp(torch.mean(torch.log(magnitude + 1e-10), dim=-2))
        arithmetic_mean = torch.mean(magnitude, dim=-2)
        ltsf = geometric_mean / (arithmetic_mean + 1e-10)

        ltsf = torch.nn.functional.avg_pool1d(ltsf, kernel_size=self.window, stride=1, padding=self.window//2).squeeze(1)
        
        return ltsf
    
    def compute_band_ratio(self,signal, band_start, band_end):
        # Compute the magnitude spectrum
        magnitude = torch.abs(signal).pow(2)
        
        # Compute the band ratio
        band_energy = torch.sum(magnitude[:, :, band_start:band_end, :], dim=-2)
        total_energy = torch.sum(magnitude, dim=-2)
        band_ratio = band_energy / (total_energy + 1e-10)
        
        return band_ratio
    
    def detect_voice_activity(self, signal, alpha, beta, band_start=9, band_end=108,
                              plot_output=True, streaming=False,
                              ema=0.9, t_low=0.4, t_high=0.6, switch_margin=0.05):

        band_ratio = self.compute_band_ratio(signal, band_start, band_end)

        # Same flatness measure, but smoothed with past frames only (causal).
        ltsf = self.compute_ltsf(signal)       
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

        score = torch.zeros(B, C, device=device)                      # running per-channel score (EMA)
        selected = torch.zeros(B, T, dtype=torch.long, device=device)
        vad = torch.zeros(B, T, device=device)
        prev = torch.zeros(B, dtype=torch.long, device=device)        # VAD hysteresis memory
        cur_channel = torch.zeros(B, dtype=torch.long, device=device) # sticky channel memory

        for t in torch.arange(T):
            score = ema * score + (1 - ema) * decision[:, :, t]       # update running score
            best = score.argmax(dim=1)                                # best channel this frame
            if t == 0:
                cur_channel = best                                   # lock onto the best at start
            else:
                # sticky: only switch if the best channel beats the held one by a margin
                switch = score[idx, best] > score[idx, cur_channel] + switch_margin
                cur_channel = torch.where(switch, best, cur_channel)
            selected[:, t] = cur_channel

            d = decision[idx, cur_channel, t]                        # chosen channel's score
            state = prev.clone()
            state[d > t_high] = 1                                     # cross high -> speech
            state[d < t_low] = 0                                      # cross low  -> noise (else hold)
            vad[:, t] = state.to(vad.dtype)
            prev = state

        return ltsf, band_ratio, score, selected, vad, decision


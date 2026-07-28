#RESEARCH
#Open Access
#Efficient voice activity detection algorithm
#using long-term spectral flatness measure

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

class vad_lps:
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
    
    def detect_voice_activity(self, signal, alpha, beta, band_start=9, band_end=108):
        ltsf = self.compute_ltsf(signal)
        band_ratio = self.compute_band_ratio(signal, band_start, band_end)
        
        # Voice activity detection based on LTSF and band ratio thresholds
        decision = alpha * (1-ltsf) + beta * band_ratio
        
        channel_score = decision.mean(dim=-1)

        selected_channel = channel_score.argmax(dim=1)
 
        return selected_channel
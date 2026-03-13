import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        device = y_true.device
        window = torch.hann_window(512).to(device)

        pred_stft = torch.stft(y_pred, 512, 256, 512, window, return_complex=True)
        true_stft = torch.stft(y_true, 512, 256, 512, window, return_complex=True)

        pred_mag = torch.abs(pred_stft).clamp(1e-8)
        true_mag = torch.abs(true_stft).clamp(1e-8)

        pred_stft_c = pred_stft / pred_mag**0.7
        true_stft_c = true_stft / true_mag**0.7

        real_loss = nn.MSELoss()(pred_stft_c.real, true_stft_c.real)
        imag_loss = nn.MSELoss()(pred_stft_c.imag, true_stft_c.imag)
        mag_loss = nn.MSELoss()(pred_mag**0.3, true_mag**0.3)

        y_norm = (torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8))

        sisnr = (-2 * torch.log10(torch.norm(y_norm, dim=-1, keepdim=True) / torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(1e-8) + 1e-8).mean())

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr


if __name__ == "__main__":
    pred = torch.randn(1, 16000)
    true = torch.randn(1, 16000)

    loss_func = HybridLoss()
    loss = loss_func(pred, true)
    print(loss)

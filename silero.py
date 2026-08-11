import torch
from silero_vad import load_silero_vad

_WINDOW = 512  
_MODEL = None


def _get_model(device):
    global _MODEL
    if _MODEL is None:
        _MODEL = load_silero_vad()
    return _MODEL.to(device)


@torch.no_grad()
def speech_prob_series(wav, fs=16000):
    model = _get_model(wav.device)
    window = _WINDOW if fs == 16000 else _WINDOW // 2
    model.reset_states()

    n_win = wav.shape[-1] // window
    if n_win == 0:
        return torch.zeros(wav.shape[0], 0, device=wav.device)
    return torch.stack(
        [model(wav[:, i * window:(i + 1) * window], fs) for i in range(n_win)], dim=1
    )


def speech_prob(wav, fs=16000):
    probs = speech_prob_series(wav, fs)
    if probs.shape[1] == 0:
        return torch.zeros(wav.shape[0], device=wav.device)
    return probs.mean(dim=1)


def select_channel(chan0_wav, chan1_wav, fs=16000): 
    prob0 = speech_prob(chan0_wav, fs)
    prob1 = speech_prob(chan1_wav, fs)
    return torch.where(prob0 < prob1, 1, 0)


def main():
    import os
    import matplotlib.pyplot as plt
    import soundfile as sf
    from gtcrn_iva_silero import multi_channel_stft, fd_wpe, auxiva

    input_dir = "test/in"
    output_dir = "test/out_silero"
    os.makedirs(output_dir, exist_ok=True)

    n_fft, hop_len, win_len, fs = 512, 256, 512, 16000
    window = torch.hann_window(win_len)

    wav_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in input_dir: {input_dir}")

    for wav_name in wav_files:
        in_path = os.path.join(input_dir, wav_name)
        base_name, _ = os.path.splitext(wav_name)
        out_path = os.path.join(output_dir, wav_name)
        try:
            signal, sr = sf.read(in_path)
            wav = torch.from_numpy(signal.T).float().unsqueeze(0)  # (B=1, C, L)

            spec_orig = multi_channel_stft(
                wav.transpose(1, 2), n_fft, hop_len, win_len, window, onesided=True
            )
            spec_drb = fd_wpe(spec_orig, rt60=0.3, shift=hop_len, D=2, fs=fs, num_iter=1)
            spec_2ch = auxiva(spec_drb.transpose(1, 3), n_iter=10).transpose(1, 3)

            chan0 = torch.istft(spec_2ch[:, 0], n_fft, hop_len, win_len, window, length=wav.shape[-1])
            chan1 = torch.istft(spec_2ch[:, 1], n_fft, hop_len, win_len, window, length=wav.shape[-1])

            out_iva_0 = chan0
            out_iva_1 = chan1
            sf.write(os.path.join(output_dir, f"{base_name}_iva_ch0.wav"), out_iva_0[0].numpy(), sr)
            sf.write(os.path.join(output_dir, f"{base_name}_iva_ch1.wav"), out_iva_1[0].numpy(), sr)

            prob0 = speech_prob_series(chan0, fs)[0]  # (n_win,)
            prob1 = speech_prob_series(chan1, fs)[0]
            selected = int(select_channel(chan0, chan1, fs=fs)[0].item())

            out_audio = (chan0 if selected == 0 else chan1)[0].numpy()
            sf.write(out_path, out_audio, sr)
            print(f"[OK] {in_path} -> {out_path} (selected ch{selected})")

            fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
            axs[0].plot(chan0[0].numpy(), label='ch0')
            axs[0].plot(chan1[0].numpy(), label='ch1', alpha=0.7)
            axs[0].set_title('IVA-separated channels')
            axs[0].legend(loc='upper right')

            axs[1].plot(prob0.numpy(), label='ch0 speech prob')
            axs[1].plot(prob1.numpy(), label='ch1 speech prob')
            axs[1].set_title('Silero VAD speech probability')
            axs[1].legend(loc='upper right')

            axs[2].axhline(selected, color='r')
            axs[2].set_ylim(-0.5, 1.5)
            axs[2].set_yticks([0, 1])
            axs[2].set_title(f'Selected channel: ch{selected}')

            fig.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_vad_silero.png'), dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"[FAILED] {in_path}: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import argparse

import torch
import soundfile as sf

from gtcrn_iva import GTCRN_IVA
import pdb


##python infer.py --input_dir <in> --output_dir <out> --checkpoint <ckpt> --aux_info <s/sn> --feature <lps/complex> --masking <mask1/mask2> --encoder <single/dual> --device <cpu/0>--suffix <ssfx>

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for GTCRN_IVA")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input noisy wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save enhanced wav files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model_0121.tar",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_enhanced",
        help="Suffix added to output filename before .wav",
    )
    parser.add_argument(
        "--aux_info",
        type=str,
        default="sn",
        choices=["s", "sn"],
        help="'s' (speech only) or 'sn' (speech & noise)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="lps",
        choices=["lps", "complex"],
        help="'lps' (log-power) or 'complex' (real-imag)",
    )
    parser.add_argument(
        "--masking",
        type=str,
        default="mask2",
        choices=["mask1", "mask2"],
        help="'mask1' (IVA output) or 'mask2' (raw noisy input)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="single",
        choices=["single", "dual"],
        help="'single' or 'dual'",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device, **model_kwargs) -> GTCRN_IVA:
    model = GTCRN_IVA(**model_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        cfg = ", ".join(f"{k}={v}" for k, v in model_kwargs.items())
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not match the requested config ({cfg}).\n"
            "The released checkpoint was trained as ID 6 (aux_info=sn, feature=lps, "
            "masking=mask2, encoder=single). Only --masking can be swapped at inference; "
            "changing --feature, --aux_info, or --encoder alters the architecture and "
            "requires a checkpoint trained with that exact config.\n"
            f"Original error: {e}"
        ) from e
    model.eval()
    return model


def enhance_one_file(model: GTCRN_IVA, wav_path: str, out_path: str, device: torch.device):
    noisy, fs = sf.read(wav_path, dtype="float32")
    if fs != 16000:
        raise ValueError(f"Expected 16000 Hz, but got {fs} for file: {wav_path}")

    if noisy.ndim == 1:
        input_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
    else:
        input_tensor = torch.from_numpy(noisy.T).unsqueeze(0).to(device)
    
    print(f"  Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    try:
        with torch.inference_mode():
            enh = model(input_tensor).cpu().numpy().squeeze()
    except Exception as e:
        print(f"  Error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    sf.write(out_path, enh, fs)


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(
        args.checkpoint,
        device,
        aux_info=args.aux_info,
        feature=args.feature,
        masking=args.masking,
        encoder=args.encoder,
    )

    wav_files = sorted(
        [
            f for f in os.listdir(args.input_dir)
            if f.lower().endswith(".wav")
        ]
    )

    if not wav_files:
        raise FileNotFoundError(f"No wav files found in input_dir: {args.input_dir}")

    for wav_name in wav_files:
        in_path = os.path.join(args.input_dir, wav_name)
        base_name, ext = os.path.splitext(wav_name)
        out_name = f"{base_name}{args.suffix}{ext}"
        out_path = os.path.join(args.output_dir, out_name)

        try:
            enhance_one_file(model, in_path, out_path, device)
            print(f"[OK] {in_path} -> {out_path}")
        except Exception as e:
            print(f"[FAILED] {in_path}: {e}")


if __name__ == "__main__":
    main()
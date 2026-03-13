# H-GTCRN
This repository is the official implementation of the Interspeech2025 paper: A Lightweight Hybrid Dual Channel Speech Enhancement System under Low-SNR Conditions. For more details, please refer to the [ISCA Archive](https://www.isca-archive.org/interspeech_2025/wang25h_interspeech.html).

| ![The framework of our proposed system.](./figures/model.png) |
|:---------------------:|
| **Figure 1:** The framework of our proposed system. |

## 🔥 News
- [**2026-3-13**] The model implementation and pre-trained checkpoint are released.
- [**2025-8-17**] The paper is uploaded to [ISCA Archive](https://www.isca-archive.org/interspeech_2025/wang25h_interspeech.html).
- [**2025-5-25**] The paper is uploaded to arxiv [![arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.19597).

## Inference
To run inference on audio files, use:

```bash
python infer.py --input_dir <input_dir> --output_dir <output_dir> --checkpoint <checkpoint> --device <device> --suffix <suffix>
```

## Audio samples
The directory structure of the audio samples is shown below:
```markdown
    samples
    ├── Samples1
    |   ├── Samples1_clean.wav
    |   ├── Samples1_noisy.wav
    |   ├── Samples1_IVA.wav
    |   ├── Samples1_GTCRN.wav
    |   ├── Samples1_DC_GTCRN.wav
    |   └── Samples1_Proposed.wav
    | ...
    └── Samples3
        ├── Samples3_clean.wav
        ├── Samples3_noisy.wav
        ├── Samples3_IVA.wav
        ├── Samples3_GTCRN.wav
        ├── Samples3_DC_GTCRN.wav
        └── Samples3_Proposed.wav
```

## Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{wang2025lightweight,
  title={A Lightweight Hybrid Dual Channel Speech Enhancement System under Low-SNR Conditions},
  author={Wang, Zheng and Rong, Xiaobin and Sun, Yu and Sun, Tianchi and Lin, Zhibin and Lu, Jing},
  booktitle={Proc. Interspeech 2025},
  pages={1178--1182},
  year={2025}
}
```

## Credits
We gratefully acknowledge the following resources that made this project possible:
- [GTCRN](https://github.com/Xiaobin-Rong/gtcrn): SOTA lightweight speech enhancement model architecture.
- [SE-train](https://github.com/Xiaobin-Rong/SEtrain): Excellent training code template for DNN-based speech enhancement.
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)

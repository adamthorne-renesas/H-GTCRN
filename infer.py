import os
import torch
import soundfile as sf
from gtcrn_iva import GTCRN_IVA

# load model
device = torch.device("cuda:0")
model = GTCRN_IVA().to(device)
checkpoint = torch.load(os.path.join("checkpoints", "best_model_0121.tar"), map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# load data
noisy, fs = sf.read(os.path.join("samples", "noisy", "003_noisy.wav"), dtype="float32")
assert fs == 16000
input = torch.FloatTensor(noisy.T).unsqueeze(0).to(device)

# inference
with torch.inference_mode():
    enh = model(input).cpu().numpy().squeeze()

# save enhanced wav
sf.write(os.path.join("samples", "enhanced", "003_enhanced.wav"), enh, fs)

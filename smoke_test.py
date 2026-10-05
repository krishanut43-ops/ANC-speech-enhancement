"""End-to-end sanity checks (CPU, no datasets needed):  python -m anc.smoke_test

1. STFT/iSTFT perfect reconstruction
2. STRICT CAUSALITY: perturbing frames >= t0 must not change outputs < t0
3. Streaming (stateful, frame-by-frame OLA) == offline whole-utterance output
4. Dataset: shapes, finite values, measured SNR vs requested
5. Composite loss + backward pass
"""
import os
import tempfile

import numpy as np
import soundfile as sf
import torch

from .config import AudioConfig, ModelConfig
from .data import DefenceNoiseSpeechDataset, active_rms, energy_vad, rms, vad_to_samples
from .losses import CompositeLoss
from .model import build_model
from .stft import STFT
from .streaming import StreamingEnhancer


def fake_speech(sr, dur, seed):
    r = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    env = np.clip(np.sin(2 * np.pi * 2.5 * t + r.uniform(0, 6)) + 0.3, 0, 1)
    f0 = 110 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    return (env * sum(np.sin(k * ph) / k for k in range(1, 8)) * 0.2).astype(np.float32)


def main():
    torch.manual_seed(0)
    a = AudioConfig()
    model = build_model(ModelConfig(), a).eval()
    stft = STFT(a.n_fft, a.hop)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model params {n_par/1e6:.2f} M | algorithmic latency {a.algorithmic_latency_ms:.0f} ms | hop {a.hop_ms:.0f} ms")

    x = torch.randn(1, a.sr * 2) * 0.1
    rec = stft.synthesis(stft.analysis(x), x.shape[-1])
    e = (rec - x).abs().max().item()
    print(f"[1] STFT reconstruction error {e:.2e}")
    assert e < 1e-5

    spec = stft.analysis(x)
    t0 = 100
    with torch.no_grad():
        y1, _ = model(spec)
        spec2 = spec.clone()
        spec2[..., t0:] += torch.randn_like(spec2[..., t0:])
        y2, _ = model(spec2)
    d_past = (y1[..., :t0] - y2[..., :t0]).abs().max().item()
    d_future = (y1[..., t0:] - y2[..., t0:]).abs().max().item()
    print(f"[2] causality: change before t0 = {d_past:.2e} (must be ~0), after t0 = {d_future:.2e}")
    assert d_past < 1e-6 and d_future > 1e-4

    off = stft.synthesis(y1, x.shape[-1])[0]
    st = StreamingEnhancer(model, a).process_file(x[0])
    d = (off - st).abs().max().item()
    print(f"[3] streaming vs offline max |diff| = {d:.2e}")
    assert d < 1e-4

    with tempfile.TemporaryDirectory() as td:
        files = []
        for i in range(3):
            p = os.path.join(td, f"s{i}.wav")
            sf.write(p, fake_speech(a.sr, 6, i), a.sr)
            files.append(p)
        ds = DefenceNoiseSpeechDataset(files, {}, a, epoch_len=8, deterministic=True, p_clip=0.0, p_reverb=0.0,
                                       gain_db_range=(0.0, 0.0), p_impulse=0.0, p_second_noise=0.0)
        errs = []
        for i in range(8):
            s = ds[i]
            assert s["noisy"].shape == s["clean"].shape == (a.segment_len,)
            assert torch.isfinite(s["noisy"]).all() and s["noisy"].abs().max() <= 1.0
            mask = vad_to_samples(energy_vad(s["clean"], a.sr), a.segment_len, int(a.sr * 0.02))
            noise = s["noisy"] - s["clean"]
            meas = 20 * torch.log10(active_rms(s["clean"], mask) / rms(noise)).item()
            errs.append(abs(meas - s["snr"].item()))
        print(f"[4] dataset ok; |measured - requested SNR| mean {np.mean(errs):.2f} dB, max {np.max(errs):.2f} dB")
        assert np.max(errs) < 1.0

        batch = [ds[i] for i in range(2)]
        noisy = torch.stack([b["noisy"] for b in batch])
        clean = torch.stack([b["clean"] for b in batch])

    model.train()
    crit = CompositeLoss(compress=a.compress)
    est_spec, _ = model(stft.analysis(noisy))
    est = stft.synthesis(est_spec, noisy.shape[-1])
    loss, parts = crit(est, clean, est_spec, stft.analysis(clean))
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[5] loss {loss.item():.3f} parts { {k: round(v.item(), 3) for k, v in parts.items()} } grad-sum {g:.1f}")
    assert torch.isfinite(loss) and g > 0
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

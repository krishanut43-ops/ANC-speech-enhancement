"""Module 4: frame-by-frame streaming inference (overlap-add) + hybrid NLMS post-filter.

Data flow per hop (128 samples = 8 ms):
    chunk -> [in_buf: last 256 samples] -> sqrt-Hann rFFT -> CRN (stateful: conv caches + GRU h)
          -> complex mask applied -> irFFT -> sqrt-Hann -> overlap-add -> emit 128 samples
The emitted block is the finished signal for the PREVIOUS chunk, so the output stream lags the
input by one hop. Worst-case algorithmic latency of any sample is the window length, 16 ms.

Hybrid ANC stage (optional): NLMS adaptive noise canceller on the enhanced signal.
    e[n] = d[n] - w^T x[n],   w <- w + mu * e[n] x[n] / (eps + ||x||^2)
  d = enhanced primary-mic signal, x = reference-mic noise (aligned to d by one hop).
  Without a reference mic, x = (delayed noisy input - enhanced) is the model's own noise estimate;
  this pseudo-reference is correlated with speech leakage, so keep mu small.
  Adaptation is frozen when the chunk looks like speech (double-talk protection), and the
  output falls back to the un-filtered signal if the filter ever increases energy.
"""
import argparse
import time

import numpy as np
import soundfile as sf
import torch

from .model import load_model
from .stft import STFT


class StreamingEnhancer:
    def __init__(self, model, acfg, device="cpu"):
        self.model, self.acfg, self.device = model.to(device).eval(), acfg, torch.device(device)
        self.stft = STFT(acfg.n_fft, acfg.hop).to(device)
        self.hop, self.n = acfg.hop, acfg.n_fft
        self.reset()

    def reset(self):
        self.state = self.model.init_state(1, self.device)
        self.in_buf = torch.zeros(1, self.n, device=self.device)
        self.ola = torch.zeros(1, self.n, device=self.device)

    @torch.inference_mode()
    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk (hop,) -> (hop,) enhanced samples of the previous chunk (one-hop lag)."""
        x = chunk.to(self.device).view(1, -1)
        self.in_buf = torch.cat([self.in_buf[:, self.hop:], x], 1)           # (1, n_fft)
        spec = self.stft.analysis_frame(self.in_buf)                          # (1, 2, F, 1)
        enh, self.state = self.model(spec, self.state)                        # (1, 2, F, 1)
        self.ola = self.ola + self.stft.synthesis_frame(enh)                  # (1, n_fft)
        out = self.ola[:, :self.hop].clone()
        self.ola = torch.cat([self.ola[:, self.hop:], torch.zeros_like(out)], 1)
        return out[0]

    def process_file(self, wav: torch.Tensor) -> torch.Tensor:
        """Run a whole signal through the streaming path (for testing / offline use)."""
        self.reset()
        L, hop = wav.numel(), self.hop
        x = torch.cat([wav, torch.zeros((-L) % hop + hop)])                   # pad + 1 flush chunk
        outs = [self.process_chunk(x[i:i + hop]) for i in range(0, x.numel(), hop)]
        return torch.cat(outs)[hop:hop + L].cpu()                             # drop the one-hop lag


class NLMSFilter:
    def __init__(self, taps=64, mu=0.05, eps=1e-6):
        self.taps, self.mu, self.eps = taps, mu, eps
        self.w = np.zeros(taps, np.float32)
        self.buf = np.zeros(taps, np.float32)

    def process(self, d: np.ndarray, x: np.ndarray, adapt: bool = True) -> np.ndarray:
        out = np.empty_like(d)
        w, buf = self.w, self.buf
        for n in range(len(d)):
            buf[1:] = buf[:-1]
            buf[0] = x[n]
            e = d[n] - w @ buf
            if adapt:
                w += (self.mu * e / (self.eps + buf @ buf)) * buf
            out[n] = e
        return out


class HybridANC:
    """AI enhancement followed by an optional NLMS residual canceller. Output lags input by one hop."""

    def __init__(self, enhancer: StreamingEnhancer, nlms: NLMSFilter = None, speech_gate_db: float = -45.0):
        self.enh, self.nlms, self.gate = enhancer, nlms, speech_gate_db
        self.prim_prev = np.zeros(enhancer.hop, np.float32)
        self.ref_prev = np.zeros(enhancer.hop, np.float32)

    def process_chunk(self, primary: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        enh = self.enh.process_chunk(torch.from_numpy(primary)).cpu().numpy()
        if reference is not None:                                   # physical reference mic, delayed to align
            x, self.ref_prev = self.ref_prev, reference.copy()
        else:                                                       # pseudo reference from model residual
            x = self.prim_prev - enh
        self.prim_prev = primary.copy()
        if self.nlms is None:
            return enh
        level_db = 10 * np.log10(np.mean(enh ** 2) + 1e-12)
        out = self.nlms.process(enh, x, adapt=level_db < self.gate)  # freeze adaptation during speech
        return out if np.mean(out ** 2) <= np.mean(enh ** 2) else enh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", help="optional reference-mic wav (same length, same sr)")
    ap.add_argument("--nlms", action="store_true")
    ap.add_argument("--taps", type=int, default=64)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, acfg = load_model(a.ckpt, a.device)
    wav, sr = sf.read(a.input, dtype="float32")
    wav = wav.mean(1) if wav.ndim > 1 else wav
    assert sr == acfg.sr, f"expected {acfg.sr} Hz, got {sr}"
    ref = None
    if a.reference:
        ref, _ = sf.read(a.reference, dtype="float32")
        ref = ref.mean(1) if ref.ndim > 1 else ref
    hop, L = acfg.hop, len(wav)
    pad = (-L) % hop + hop
    wav_p = np.concatenate([wav, np.zeros(pad, np.float32)])
    ref_p = np.concatenate([ref, np.zeros(pad, np.float32)]) if ref is not None else None

    hybrid = HybridANC(StreamingEnhancer(model, acfg, a.device), NLMSFilter(a.taps, a.mu) if a.nlms else None)
    outs, times = [], []
    for i in range(0, len(wav_p), hop):
        t0 = time.perf_counter()
        outs.append(hybrid.process_chunk(wav_p[i:i + hop], ref_p[i:i + hop] if ref_p is not None else None))
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    y = np.concatenate(outs)[hop:hop + L]
    sf.write(a.output, y, sr)
    t = np.array(times[5:])                                          # skip warm-up chunks
    print(f"algorithmic latency {acfg.algorithmic_latency_ms:.1f} ms | compute/chunk mean {t.mean():.2f} ms "
          f"p99 {np.percentile(t, 99):.2f} ms max {t.max():.2f} ms | budget {acfg.hop_ms:.1f} ms "
          f"| real-time factor {t.mean() / acfg.hop_ms:.2f}")


if __name__ == "__main__":
    main()

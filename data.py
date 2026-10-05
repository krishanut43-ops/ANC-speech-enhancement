"""Module 1: scalable on-the-fly synthetic dataset for defence noise suppression.

Every __getitem__ builds a fresh (noisy, clean) pair, so the effective dataset size is
unbounded and nothing is stored on disk.

Signal-level conventions (important for correct SNR):
  * Clean speech is VAD-gated: a frame is "active" if its energy is within `rel_db` of the
    loudest frame (and above an absolute floor). Speech level = RMS over ACTIVE samples only
    (P.56-style), so pauses do not distort the SNR definition.
  * Speech is normalised to `ref_dbfs` (default -25 dBFS active level).
  * Noise level = RMS over the whole segment. It is scaled so that
        10*log10(P_speech_active / P_noise) = SNR   (SNR ~ U(snr_range)).
  * Random global gain (dynamic volume) and clipping are applied AFTER mixing, jointly to the
    noisy and clean signals (gain) or noisy only (clipping), so the target SNR is preserved.

Augmentations: RIR reverberation (speech and noise), random impulse injection, clipping,
slow time-varying noise gain, 2-source noise mixtures, random global volume.
Target = dry speech convolved with the EARLY part (first 50 ms) of the RIR if `dereverb`,
i.e. the model also learns to remove late reverberation.

Folder layout expected by `scan_noise_dir`:
    noise_root/{gunshot,artillery,helicopter,vehicle,drone,wind,siren}/*.wav|flac
Categories without files fall back to parametric synthetic noise (for smoke tests / bootstrapping;
replace with real recordings for any serious result).
"""
import glob
import math
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, lfilter, resample_poly
from torch.utils.data import Dataset

from .config import AudioConfig

# name: (sampling weight, impulsive?)  impulsive clips are placed as sparse events, others are tiled
NOISE_CATEGORIES: Dict[str, Tuple[float, bool]] = {
    "gunshot": (0.20, True),
    "artillery": (0.15, True),
    "helicopter": (0.15, False),
    "vehicle": (0.15, False),
    "drone": (0.10, False),
    "wind": (0.10, False),
    "siren": (0.15, False),
}
CATEGORY_NAMES = list(NOISE_CATEGORIES)
AUDIO_EXT = ("*.wav", "*.flac", "*.ogg")


# --------------------------------------------------------------------------- IO
def scan_files(root: str) -> List[str]:
    out = []
    for ext in AUDIO_EXT:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(out)


def scan_noise_dir(root: str) -> Dict[str, List[str]]:
    return {c: scan_files(os.path.join(root, c)) for c in NOISE_CATEGORIES
            if os.path.isdir(os.path.join(root, c))}


def load_audio(path: str, sr: int) -> torch.Tensor:
    wav, fs = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if fs != sr:
        g = math.gcd(fs, sr)
        wav = resample_poly(wav, sr // g, fs // g).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(wav))


# ------------------------------------------------------------------- level / VAD
def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).mean().clamp_min(eps).sqrt()


def energy_vad(wav: torch.Tensor, sr: int, frame_ms: int = 20,
               rel_db: float = -30.0, abs_db: float = -55.0) -> torch.Tensor:
    """Frame-level energy VAD -> bool (n_frames,). Active if within rel_db of the peak frame and > abs_db."""
    n = int(sr * frame_ms / 1000)
    T = wav.numel() // n
    e_db = 10 * torch.log10(wav[:T * n].view(T, n).pow(2).mean(1) + 1e-10)
    return e_db > max(e_db.max().item() + rel_db, abs_db)


def vad_to_samples(vad: torch.Tensor, n_samples: int, frame_len: int) -> torch.Tensor:
    m = vad.repeat_interleave(frame_len)
    if m.numel() < n_samples:
        m = torch.cat([m, torch.zeros(n_samples - m.numel(), dtype=torch.bool)])
    return m[:n_samples]


def active_rms(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return rms(x[mask]) if mask.any() else rms(x)


# ------------------------------------------------------------- reverb / impulses
def synth_rir(rng, sr: int, rt60=None, early_ms: float = 50.0):
    """Synthetic exponential-decay RIR. Returns (full, early) as numpy arrays; direct path at index 0."""
    rt60 = rt60 if rt60 is not None else rng.uniform(0.2, 0.9)
    n = int(rt60 * sr) + 1
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)      # -60 dB at rt60
    tail[: int(0.002 * sr)] = 0.0
    drr_db = rng.uniform(-2.0, 10.0)                                # direct-to-reverberant ratio
    tail *= math.sqrt(10 ** (-drr_db / 10) / (np.sum(tail ** 2) + 1e-12))
    h = tail.copy()
    h[0] = 1.0
    return h, h[: int(early_ms / 1000 * sr)]


def conv(x: torch.Tensor, h: np.ndarray) -> torch.Tensor:
    y = fftconvolve(x.numpy(), h)[: x.numel()]
    return torch.from_numpy(y.astype(np.float32))


def inject_impulses(x: torch.Tensor, rng, sr: int, max_events: int = 4) -> torch.Tensor:
    """Add 1..max_events synthetic decaying bursts (gunshot-like) with random spectral tilt."""
    base, out = float(rms(x)), x.clone()
    for _ in range(int(rng.integers(1, max_events + 1))):
        n = int(rng.uniform(0.02, 0.15) * sr)
        tau = rng.uniform(0.003, 0.03)
        b = rng.standard_normal(n) * np.exp(-np.arange(n) / sr / tau)
        a = rng.uniform(0.0, 0.8)
        b = lfilter([1 - a], [1, -a], b)
        b = b / (np.abs(b).max() + 1e-9) * base * rng.uniform(3.0, 10.0)
        pos = int(rng.integers(0, x.numel() - n))
        out[pos:pos + n] += torch.from_numpy(b.astype(np.float32))
    return out


def gain_envelope(L: int, sr: int, rng, max_db: float = 6.0, period: float = 0.5) -> torch.Tensor:
    """Slowly varying gain (e.g. a helicopter approaching / leaving)."""
    k = max(2, int(L / sr / period) + 1)
    pts = rng.uniform(-max_db, max_db, k)
    env_db = np.interp(np.linspace(0, k - 1, L), np.arange(k), pts)
    return torch.from_numpy((10 ** (env_db / 20)).astype(np.float32))


# --------------------------------------------------------- synthetic noise fallback
def _lp(x, a):
    return lfilter([1 - a], [1, -a], x)


def synth_noise(cat: str, L: int, sr: int, rng) -> torch.Tensor:
    """Parametric stand-ins so the pipeline runs without datasets. NOT a substitute for real recordings."""
    t = np.arange(L) / sr
    if cat in ("gunshot", "artillery"):
        gun = cat == "gunshot"
        x = np.zeros(L)
        for _ in range(int(rng.integers(1, 5))):
            n = int((0.15 if gun else 0.8) * sr)
            tau = (0.01 if gun else 0.12) * rng.uniform(0.5, 1.5)
            b = _lp(rng.standard_normal(n) * np.exp(-np.arange(n) / sr / tau), 0.2 if gun else 0.93)
            pos = int(rng.integers(0, L - n))
            x[pos:pos + n] += b / (np.abs(b).max() + 1e-9) * rng.uniform(0.5, 1.0)
    elif cat == "helicopter":
        f = rng.uniform(10, 25)
        am = 0.6 + 0.4 * np.sin(2 * np.pi * f * t) ** 2
        x = am * _lp(rng.standard_normal(L), 0.9) + 0.5 * sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 9))
    elif cat == "vehicle":
        f0 = rng.uniform(25, 60)
        x = sum(np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) / k for k in range(1, 11)) \
            + 0.3 * _lp(rng.standard_normal(L), 0.8)
    elif cat == "drone":
        f0 = rng.uniform(150, 450)
        ph = 2 * np.pi * (f0 * t + 3 * np.sin(2 * np.pi * 0.5 * t))
        x = sum(np.sin(k * ph) / k for k in range(1, 6)) + 0.2 * rng.standard_normal(L)
    elif cat == "wind":
        x = _lp(rng.standard_normal(L), 0.97) * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t))
    elif cat == "siren":
        f = rng.uniform(600, 900) + 250 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t)
        ph = 2 * np.pi * np.cumsum(f) / sr
        x = np.sin(ph) + 0.3 * np.sin(2 * ph)
    else:
        x = rng.standard_normal(L)
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


# -------------------------------------------------------------------- dataset
class DefenceNoiseSpeechDataset(Dataset):
    def __init__(self, speech_files: List[str], noise_files: Dict[str, List[str]],
                 audio_cfg: AudioConfig = AudioConfig(), epoch_len: int = 10000,
                 snr_range=(-10.0, 15.0), deterministic: bool = False, seed: int = 0,
                 use_synthetic_fallback: bool = True, dereverb: bool = True,
                 p_reverb=0.6, p_noise_reverb=0.4, p_impulse=0.3, p_clip=0.2, p_second_noise=0.3,
                 min_speech_ratio=0.35, ref_dbfs=-25.0, gain_db_range=(-15.0, 5.0)):
        assert len(speech_files) > 0, "no speech files"
        assert audio_cfg.segment_len % audio_cfg.hop == 0
        self.speech_files, self.noise_files, self.acfg = speech_files, noise_files, audio_cfg
        self.epoch_len, self.snr_range = epoch_len, snr_range
        self.deterministic, self.seed, self.dereverb = deterministic, seed, dereverb
        self.p = dict(reverb=p_reverb, noise_reverb=p_noise_reverb, impulse=p_impulse,
                      clip=p_clip, second=p_second_noise)
        self.min_speech_ratio, self.ref_dbfs, self.gain_db_range = min_speech_ratio, ref_dbfs, gain_db_range
        self._calls = 0

        self.cats = [c for c in NOISE_CATEGORIES if noise_files.get(c) or use_synthetic_fallback]
        if not self.cats:
            raise ValueError("no noise files found and synthetic fallback disabled")
        missing = [c for c in self.cats if not noise_files.get(c)]
        if missing:
            warnings.warn(f"synthetic noise used for categories without files: {missing}")
        w = np.array([NOISE_CATEGORIES[c][0] for c in self.cats])
        self.cat_p = w / w.sum()

    def __len__(self):
        return self.epoch_len

    def _rng(self, idx):
        if self.deterministic:
            return np.random.default_rng([self.seed, idx])
        self._calls += 1
        return np.random.default_rng([torch.initial_seed() % (2 ** 32), idx, self._calls])

    # ----- speech
    def _speech_segment(self, rng) -> torch.Tensor:
        L, sr = self.acfg.segment_len, self.acfg.sr
        best, best_ratio = None, -1.0
        for _ in range(8):                                    # VAD check with retries
            pick = lambda: self.speech_files[int(rng.integers(len(self.speech_files)))]
            wav = load_audio(pick(), sr)
            while wav.numel() < L:                            # concatenate utterances if too short
                gap = torch.zeros(int(rng.uniform(0.1, 0.4) * sr))
                wav = torch.cat([wav, gap, load_audio(pick(), sr)])
            start = int(rng.integers(0, wav.numel() - L + 1))
            seg = wav[start:start + L]
            ratio = energy_vad(seg, sr).float().mean().item()
            if ratio > best_ratio:
                best, best_ratio = seg, ratio
            if ratio >= self.min_speech_ratio:
                break
        return best

    # ----- noise
    def _noise_track(self, cat: str, rng, L: int) -> torch.Tensor:
        sr, files = self.acfg.sr, self.noise_files.get(cat)
        if not files:
            return synth_noise(cat, L, sr, rng)
        pick = lambda: files[int(rng.integers(len(files)))]
        if NOISE_CATEGORIES[cat][1]:                          # impulsive: sparse events on silence
            track = torch.zeros(L)
            for _ in range(int(rng.integers(1, 5))):
                clip = load_audio(pick(), sr)
                n = min(clip.numel(), 2 * sr)
                s0 = int(rng.integers(0, clip.numel() - n + 1))
                clip = clip[s0:s0 + n] * 10 ** (rng.uniform(-6, 6) / 20)
                pos = int(rng.integers(0, max(1, L - n)))
                seg = clip[: L - pos]
                track[pos:pos + seg.numel()] += seg
            return track
        wav = load_audio(pick(), sr)                          # continuous: tile + random crop
        if wav.numel() < L:
            wav = wav.repeat(math.ceil(L / wav.numel()) + 1)
        s0 = int(rng.integers(0, wav.numel() - L + 1))
        return wav[s0:s0 + L].clone()

    def _noise_mixture(self, rng, L):
        sr = self.acfg.sr
        n_src = min(2 if rng.random() < self.p["second"] else 1, len(self.cats))
        idxs = rng.choice(len(self.cats), size=n_src, replace=False, p=self.cat_p)
        mix = torch.zeros(L)
        for k, ci in enumerate(idxs):
            tr = self._noise_track(self.cats[ci], rng, L)
            if rng.random() < self.p["noise_reverb"]:
                tr = conv(tr, synth_rir(rng, sr)[0])
            tr = tr * gain_envelope(L, sr, rng)               # dynamic volume of the source
            tr = tr / rms(tr)                                 # unit RMS before mixing
            if k > 0:
                tr = tr * 10 ** (rng.uniform(-10, 0) / 20)    # secondary source 0..-10 dB
            mix += tr
        if rng.random() < self.p["impulse"]:
            mix = inject_impulses(mix, rng, sr)
        return mix, CATEGORY_NAMES.index(self.cats[idxs[0]])

    # ----- main
    def __getitem__(self, idx):
        rng, sr, L = self._rng(idx), self.acfg.sr, self.acfg.segment_len
        speech = self._speech_segment(rng)                                    # dry (L,)
        frame_len = int(sr * 0.02)
        mask = vad_to_samples(energy_vad(speech, sr), L, frame_len)

        if rng.random() < self.p["reverb"]:                                   # RIR reverberation
            h, h_early = synth_rir(rng, sr)
            rev = conv(speech, h)
            target = conv(speech, h_early) if self.dereverb else rev
        else:
            rev = target = speech

        g = 10 ** (self.ref_dbfs / 20) / active_rms(target, mask)             # gain normalisation
        rev, target = rev * g, target * g

        noise, cat = self._noise_mixture(rng, L)
        snr = float(rng.uniform(*self.snr_range))
        noise = noise * (active_rms(target, mask) / (rms(noise) * 10 ** (snr / 20)))
        noisy = rev + noise

        if rng.random() < self.p["clip"]:                                     # mic saturation (noisy only)
            thr = float(rng.uniform(0.3, 0.9) * noisy.abs().max())
            noisy = noisy.clamp(-thr, thr)

        vg = 10 ** (rng.uniform(*self.gain_db_range) / 20)                    # joint volume change
        noisy, clean = noisy * vg, target * vg
        peak = max(noisy.abs().max(), clean.abs().max())
        if peak > 0.99:
            noisy, clean = noisy * (0.99 / peak), clean * (0.99 / peak)
        return dict(noisy=noisy.float(), clean=clean.float(),
                    snr=torch.tensor(snr), cat=torch.tensor(cat))

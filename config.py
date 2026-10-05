"""Shared configuration for the causal speech-enhancement stack.

Latency budget (algorithmic, zero look-ahead):
    A frame is only complete after `n_fft` samples have arrived, so the
    algorithmic delay of an STFT/overlap-add system equals the window length:
        n_fft / sr = 256 / 16000 = 16 ms  (< 20 ms target)
    The hop (8 ms) is the granularity at which the model is invoked; the model
    must finish each hop in well under 8 ms to run in real time.
"""
from dataclasses import dataclass


@dataclass
class AudioConfig:
    sr: int = 16000
    n_fft: int = 256          # 16 ms analysis window, 129 frequency bins
    hop: int = 128            # 8 ms hop (50 % overlap, sqrt-Hann => perfect reconstruction)
    compress: float = 0.3     # power-law compression exponent for the network input / loss
    segment_sec: float = 4.0  # training segment length

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def segment_len(self) -> int:
        n = int(self.sr * self.segment_sec)
        return n - n % self.hop  # must be a multiple of hop

    @property
    def algorithmic_latency_ms(self) -> float:
        return 1000.0 * self.n_fft / self.sr

    @property
    def hop_ms(self) -> float:
        return 1000.0 * self.hop / self.sr


@dataclass
class ModelConfig:
    enc_channels: tuple = (16, 32, 48, 64)  # complex channels per encoder stage
    kf: int = 5          # frequency kernel (non-causal axis, it is not time)
    kt: int = 2          # TIME kernel: current + 1 past frame (causal)
    hidden: int = 256    # GRU width
    gru_layers: int = 2

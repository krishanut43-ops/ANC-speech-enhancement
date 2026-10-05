"""Causal STFT / iSTFT with an explicit frame API shared by training and streaming.

Convention: spectra are real tensors of shape (B, 2, F, T) with
channel 0 = real part and channel 1 = imaginary part, F = n_fft//2 + 1.

The window is sqrt-Hann (periodic). With 50 % overlap, w^2 sums to 1, so
analysis followed by synthesis reconstructs the input exactly. The signal is
zero-padded by one hop on both sides so that every real sample is covered by
two frames; this is exactly the state a streaming system has at start-up
(an input buffer of zeros), which makes offline and streaming outputs match.
"""
import torch
import torch.nn.functional as F
from torch import nn


class STFT(nn.Module):
    def __init__(self, n_fft: int = 256, hop: int = 128):
        super().__init__()
        assert n_fft == 2 * hop, "sqrt-Hann perfect reconstruction assumes 50 % overlap"
        self.n_fft, self.hop = n_fft, hop
        win = torch.hann_window(n_fft, periodic=True).sqrt()
        self.register_buffer("win", win, persistent=False)  # (n_fft,)

    # ---------------------------------------------------------------- offline
    def analysis(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, L) with L % hop == 0  ->  spec (B, 2, F, T), T = L/hop + 1."""
        B, L = wav.shape
        assert L % self.hop == 0, f"length {L} must be a multiple of hop {self.hop}"
        x = F.pad(wav, (self.hop, self.hop))                   # (B, L + 2*hop)
        frames = x.unfold(-1, self.n_fft, self.hop)            # (B, T, n_fft)
        Z = torch.fft.rfft(frames * self.win, dim=-1)          # (B, T, F) complex
        return torch.stack([Z.real, Z.imag], 1).transpose(2, 3).contiguous()  # (B,2,F,T)

    def synthesis(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """spec (B, 2, F, T) -> wav (B, length). Windowed overlap-add."""
        Z = torch.complex(spec[:, 0], spec[:, 1])              # (B, F, T)
        frames = torch.fft.irfft(Z, n=self.n_fft, dim=1)       # (B, n_fft, T)
        frames = frames * self.win[None, :, None]
        out = F.fold(frames, output_size=(1, length + 2 * self.hop),
                     kernel_size=(1, self.n_fft), stride=(1, self.hop))  # (B,1,1,L+2hop)
        return out[:, 0, 0, self.hop:self.hop + length]

    # -------------------------------------------------------------- streaming
    def analysis_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """One time-domain frame (B, n_fft) -> spec (B, 2, F, 1)."""
        Z = torch.fft.rfft(frame * self.win, dim=-1)           # (B, F)
        return torch.stack([Z.real, Z.imag], 1).unsqueeze(-1)

    def synthesis_frame(self, spec: torch.Tensor) -> torch.Tensor:
        """spec (B, 2, F, 1) -> windowed time frame (B, n_fft), ready for overlap-add."""
        Z = torch.complex(spec[:, 0, :, 0], spec[:, 1, :, 0])  # (B, F)
        return torch.fft.irfft(Z, n=self.n_fft, dim=-1) * self.win

"""Module 2: Causal Complex CRN (convolutional-recurrent network) with a complex ratio mask.

Layout of "complex features": a tensor (B, 2C, F, T) whose first C channels are the
real parts and last C channels the imaginary parts. A complex convolution is
    (Wr + jWi) * (xr + jxi) = (Wr*xr - Wi*xi) + j(Wr*xi + Wi*xr)
implemented with two real convolutions.

Causality
    * TIME axis: every conv uses kernel kt with LEFT-only context (cache of the last
      kt-1 frames, zeros at start-up). Decoder transposed convs are time-pointwise.
    * The recurrent core is a unidirectional GRU.
    * BatchNorm runs with fixed statistics at inference (eval()).
    * FREQUENCY axis convs are symmetric - frequency is not time, so this does not
      violate causality.
    => output frame t depends only on input frames <= t. `smoke_test.py` verifies it.

Streaming
    forward(spec, state) takes ANY number of frames T (1 for real-time) and returns the
    new state, so the same code path serves training (state=None, whole utterance),
    streaming inference, and ONNX/TensorRT export.

Shapes (n_fft=256 -> F=129, defaults):
    input  spec  (B, 2, 129, T)       raw noisy STFT (real, imag)
    enc1         (B, 32, 65, T)       2*16 channels
    enc2         (B, 64, 33, T)
    enc3         (B, 96, 17, T)
    enc4         (B, 128, 9, T)       bottleneck feature map
    flatten      (B, T, 1152)         2*64*9
    GRU          (B, T, 256)
    dec -> mask  (B, 2, 129, T)       complex mask (real, imag)
    output       (B, 2, 129, T)       enhanced STFT = mask * noisy
"""
from dataclasses import asdict
from typing import List, Optional

import torch
import torch.nn as nn

from .config import AudioConfig, ModelConfig


def cat_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate two complex feature maps along the channel axis, keeping [real | imag] layout."""
    ar, ai = a.chunk(2, 1)
    br, bi = b.chunk(2, 1)
    return torch.cat([ar, br, ai, bi], 1)


def _complex_mul_conv(conv_r, conv_i, x):
    xr, xi = x.chunk(2, 1)
    return torch.cat([conv_r(xr) - conv_i(xi), conv_r(xi) + conv_i(xr)], 1)


class CausalComplexConv2d(nn.Module):
    """Complex conv, stride along frequency only, causal along time.

    x     (B, 2*cin, F, T)
    cache (B, 2*cin, F, kt-1) or None (zeros)
    out   (B, 2*cout, F_out, T), new_cache
    """

    def __init__(self, cin: int, cout: int, kf: int, kt: int, sf: int = 2):
        super().__init__()
        self.kt = kt
        args = dict(kernel_size=(kf, kt), stride=(sf, 1), padding=(kf // 2, 0))
        self.conv_r = nn.Conv2d(cin, cout, **args)
        self.conv_i = nn.Conv2d(cin, cout, **args)

    def forward(self, x, cache: Optional[torch.Tensor] = None):
        if cache is None:
            cache = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], self.kt - 1)
        x = torch.cat([cache, x], dim=-1)           # (B, 2cin, F, kt-1+T): left context only
        new_cache = x[..., -(self.kt - 1):]         # last kt-1 frames feed the next call
        return _complex_mul_conv(self.conv_r, self.conv_i, x), new_cache


class CausalComplexConvT2d(nn.Module):
    """Complex transposed conv, upsamples frequency x2 (F -> 2F-1), pointwise in time."""

    def __init__(self, cin: int, cout: int, kf: int = 3, sf: int = 2):
        super().__init__()
        args = dict(kernel_size=(kf, 1), stride=(sf, 1), padding=(kf // 2, 0))
        self.conv_r = nn.ConvTranspose2d(cin, cout, **args)
        self.conv_i = nn.ConvTranspose2d(cin, cout, **args)

    def forward(self, x):
        return _complex_mul_conv(self.conv_r, self.conv_i, x)


class EncBlock(nn.Module):
    def __init__(self, cin, cout, kf, kt):
        super().__init__()
        self.conv = CausalComplexConv2d(cin, cout, kf, kt)
        self.bn = nn.BatchNorm2d(2 * cout)
        self.act = nn.PReLU(2 * cout)

    def forward(self, x, cache=None):
        y, new_cache = self.conv(x, cache)
        return self.act(self.bn(y)), new_cache


class DecBlock(nn.Module):
    def __init__(self, cin, cout, last=False):
        super().__init__()
        self.conv = CausalComplexConvT2d(cin, cout)
        self.post = nn.Identity() if last else nn.Sequential(nn.BatchNorm2d(2 * cout), nn.PReLU(2 * cout))

    def forward(self, x):
        return self.post(self.conv(x))


class CausalComplexCRN(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig(), n_freq: int = 129, compress: float = 0.3):
        super().__init__()
        self.cfg, self.n_freq, self.compress = cfg, n_freq, compress
        ch = [1, *cfg.enc_channels]
        n_enc = len(cfg.enc_channels)
        self.enc = nn.ModuleList([EncBlock(ch[i], ch[i + 1], cfg.kf, cfg.kt) for i in range(n_enc)])

        # frequency size entering each encoder block, and at the bottleneck
        self.f_in, f = [], n_freq
        for _ in range(n_enc):
            self.f_in.append(f)
            f = (f + 2 * (cfg.kf // 2) - cfg.kf) // 2 + 1
        self.f_bottle = f
        fd = f
        for _ in range(n_enc):
            fd = 2 * fd - 1
        assert fd == n_freq, f"decoder would output {fd} bins, expected {n_freq}; use n_fft=256"

        feat = 2 * ch[-1] * self.f_bottle
        self.pre = nn.Sequential(nn.Linear(feat, cfg.hidden), nn.PReLU())
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, cfg.gru_layers, batch_first=True)  # unidirectional
        self.post = nn.Linear(cfg.hidden, feat)

        dec, cur = [], ch[-1]
        for i in reversed(range(n_enc)):
            cin = cur + cfg.enc_channels[i]                      # skip connection from enc block i
            cout = cfg.enc_channels[i - 1] if i > 0 else 1       # last block -> 1 complex channel (mask)
            dec.append(DecBlock(cin, cout, last=(i == 0)))
            cur = cout
        self.dec = nn.ModuleList(dec[::-1])  # dec[i] pairs with enc[i]

    # ------------------------------------------------------------------ state
    def init_state(self, batch: int = 1, device=None) -> List[torch.Tensor]:
        """[cache_0 .. cache_{n_enc-1}, h]  with cache_i (B, 2*cin_i, F_i, kt-1), h (layers, B, H)."""
        device = device or next(self.parameters()).device
        ch = [1, *self.cfg.enc_channels]
        st = [torch.zeros(batch, 2 * ch[i], self.f_in[i], self.cfg.kt - 1, device=device)
              for i in range(len(self.enc))]
        st.append(torch.zeros(self.cfg.gru_layers, batch, self.cfg.hidden, device=device))
        return st

    # ---------------------------------------------------------------- forward
    def forward(self, spec: torch.Tensor, state: Optional[List[torch.Tensor]] = None):
        """spec (B, 2, F, T) -> (enhanced spec (B, 2, F, T), new_state)."""
        n_enc = len(self.enc)
        caches = list(state[:n_enc]) if state is not None else [None] * n_enc
        h = state[n_enc] if state is not None else None

        # power-law compressed complex input: |x|^c * e^{j phase}  (tames gunshot dynamics)
        mag = torch.sqrt(spec[:, :1] ** 2 + spec[:, 1:] ** 2 + 1e-8)      # (B,1,F,T)
        x = spec * mag.pow(self.compress - 1.0)                             # (B,2,F,T)

        skips, new_caches = [], []
        for blk, c in zip(self.enc, caches):
            x, nc = blk(x, c)
            skips.append(x)
            new_caches.append(nc)

        B, C2, Fb, T = x.shape
        z = x.permute(0, 3, 1, 2).reshape(B, T, C2 * Fb)    # (B, T, 2C*F_b)
        z = self.pre(z)                                      # (B, T, H)
        z, h_new = self.gru(z, h)                            # (B, T, H), (layers, B, H)
        z = self.post(z)                                     # (B, T, 2C*F_b)
        x = z.reshape(B, T, C2, Fb).permute(0, 2, 3, 1)      # (B, 2C, F_b, T)

        for i in reversed(range(n_enc)):
            x = self.dec[i](cat_complex(x, skips[i]))        # (B, 2*cout, 2F-1, T)

        # complex ratio mask with bounded magnitude (tanh), phase preserved from network
        mr, mi = x[:, 0].float(), x[:, 1].float()                           # (B, F, T)
        norm = torch.sqrt(mr ** 2 + mi ** 2 + 1e-8)
        scale = torch.tanh(norm) / norm
        mr, mi = mr * scale, mi * scale
        sr, si = spec[:, 0], spec[:, 1]
        out = torch.stack([sr * mr - si * mi, sr * mi + si * mr], 1)       # (B, 2, F, T)
        return out, new_caches + [h_new]


def build_model(mcfg: ModelConfig = ModelConfig(), acfg: AudioConfig = AudioConfig()) -> CausalComplexCRN:
    return CausalComplexCRN(mcfg, acfg.n_freq, acfg.compress)


def checkpoint_dict(model, mcfg, acfg, **extra) -> dict:
    return dict(model=model.state_dict(), model_cfg=asdict(mcfg), audio_cfg=asdict(acfg), **extra)


def load_model(path: str, device="cpu"):
    ck = torch.load(path, map_location=device)
    mcfg, acfg = ModelConfig(**ck["model_cfg"]), AudioConfig(**ck["audio_cfg"])
    model = build_model(mcfg, acfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), acfg

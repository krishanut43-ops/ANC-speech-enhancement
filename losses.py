"""Module 3: composite loss = SI-SNR (time) + multi-resolution STFT + compressed complex spectrum.

Math
  SI-SNR:  s_t = <e,s>/||s||^2 * s ,  e_n = e - s_t ,  SI-SNR = 10 log10(||s_t||^2 / ||e_n||^2)
           (both signals zero-meaned; loss = -mean SI-SNR, invariant to output gain)
  MR-STFT: per resolution, spectral convergence ||  |S|-|S^| ||_F / || |S| ||_F
           plus log-magnitude L1  mean | log|S| - log|S^| |
  Complex compressed spectrum (phase aware, CMGAN/PHASEN style), c = 0.3:
           S_c = |S|^c e^{j angle(S)};   loss = a * MSE(|S|^c, |S^|^c) + (1-a) * MSE(Re,Im of S_c)
           Compression stops loud gunshot bins from dominating, and the real/imag term
           forces phase accuracy, not only magnitude.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """est, ref (B, L) -> SI-SNR in dB, shape (B,)."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    proj = (est * ref).sum(-1, keepdim=True) * ref / (ref.pow(2).sum(-1, keepdim=True) + eps)
    noise = est - proj
    return 10 * torch.log10(proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


class MultiResSTFTLoss(nn.Module):
    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.res = resolutions
        for i, (n_fft, _) in enumerate(resolutions):
            self.register_buffer(f"win{i}", torch.hann_window(n_fft), persistent=False)

    @staticmethod
    def _mag(x, n_fft, hop, win):
        Z = torch.stft(x, n_fft, hop, n_fft, win, return_complex=True)
        return torch.sqrt(Z.real ** 2 + Z.imag ** 2 + 1e-10)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for i, (n_fft, hop) in enumerate(self.res):
            win = getattr(self, f"win{i}")
            E, R = self._mag(est, n_fft, hop, win), self._mag(ref, n_fft, hop, win)
            sc = torch.norm(R - E, p="fro", dim=(-2, -1)) / (torch.norm(R, p="fro", dim=(-2, -1)) + 1e-8)
            lm = F.l1_loss(torch.log(E), torch.log(R))
            total = total + sc.mean() + lm
        return total / len(self.res)


def compressed_complex_loss(est_spec, ref_spec, c: float = 0.3, alpha: float = 0.3):
    """est_spec, ref_spec (B, 2, F, T) real/imag."""
    def comp(s):
        mag = torch.sqrt(s[:, :1] ** 2 + s[:, 1:] ** 2 + 1e-8)
        return s * mag.pow(c - 1.0), mag.pow(c)

    ec, em = comp(est_spec)
    rc, rm = comp(ref_spec)
    return alpha * F.mse_loss(em, rm) + (1 - alpha) * F.mse_loss(ec, rc)


class CompositeLoss(nn.Module):
    def __init__(self, w_sisnr=0.1, w_mrstft=1.0, w_cplx=1.0, compress=0.3):
        super().__init__()
        self.w = (w_sisnr, w_mrstft, w_cplx)
        self.mr = MultiResSTFTLoss()
        self.compress = compress

    def forward(self, est_wav, ref_wav, est_spec, ref_spec):
        l_si = -si_snr(est_wav, ref_wav).mean()
        l_mr = self.mr(est_wav, ref_wav)
        l_cx = compressed_complex_loss(est_spec, ref_spec, self.compress)
        total = self.w[0] * l_si + self.w[1] * l_mr + self.w[2] * l_cx
        return total, dict(si_snr=-l_si.detach(), mrstft=l_mr.detach(), cplx=l_cx.detach())

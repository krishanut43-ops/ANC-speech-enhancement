"""Module 3: training script (AMP, warmup+cosine LR, online STOI / PESQ / SNR evaluation).

Example:
    python -m anc.train --speech-root LibriSpeech/train-clean-100 \
        --val-speech-root LibriSpeech/dev-clean --noise-root noise/ \
        --out-dir runs/crn_v1 --epochs 60 --batch-size 16 --amp

Without --noise-root the parametric synthetic noise is used (smoke testing only).
Noise files are split 90/10 per category BY FILE so validation noise is never seen in training.
"""
import argparse
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import AudioConfig, ModelConfig
from .data import CATEGORY_NAMES, DefenceNoiseSpeechDataset, scan_files, scan_noise_dir
from .losses import CompositeLoss
from .model import build_model, checkpoint_dict
from .stft import STFT


def split_noise(noise, val_frac=0.1, seed=0):
    rnd, tr, va = random.Random(seed), {}, {}
    for c, files in noise.items():
        files = sorted(files)
        rnd.shuffle(files)
        k = max(1, int(len(files) * val_frac)) if len(files) > 1 else 0
        va[c], tr[c] = (files[:k] or files), (files[k:] or files)
    return tr, va


def run_model(model, stft, noisy, amp=False):
    """noisy (B, L) -> (enhanced spec (B,2,F,T) fp32, enhanced wav (B, L))."""
    spec = stft.analysis(noisy)                                   # STFT in fp32, outside autocast
    with torch.autocast(device_type=noisy.device.type, dtype=torch.float16, enabled=amp):
        est_spec, _ = model(spec)
    est_spec = est_spec.float()
    return est_spec, stft.synthesis(est_spec, noisy.shape[-1])


def snr_db(ref, est):
    return 10 * np.log10((ref ** 2).sum() / (((ref - est) ** 2).sum() + 1e-10) + 1e-10)


@torch.no_grad()
def evaluate(model, stft, criterion, loader, device, sr, max_items):
    from pesq import pesq          # imported lazily: only needed for evaluation
    from pystoi import stoi

    model.eval()
    rec, losses, seen = defaultdict(lambda: defaultdict(list)), [], 0
    for b in loader:
        noisy, clean = b["noisy"].to(device), b["clean"].to(device)
        est_spec, est = run_model(model, stft, noisy)
        loss, _ = criterion(est, clean, est_spec, stft.analysis(clean))
        losses.append(loss.item())
        for i in range(noisy.shape[0]):
            c, n, e = (t[i].cpu().numpy().astype(np.float64) for t in (clean, noisy, est))
            m = {"snr_in": snr_db(c, n), "snr_out": snr_db(c, e), "stoi_in": stoi(c, n, sr),
                 "stoi_out": stoi(c, e, sr)}
            try:
                m["pesq_in"], m["pesq_out"] = pesq(sr, c, n, "wb"), pesq(sr, c, e, "wb")
            except Exception:                                      # e.g. NoUtterancesError
                m["pesq_in"] = m["pesq_out"] = float("nan")
            m["snr_gain"] = m["snr_out"] - m["snr_in"]
            for k, v in m.items():
                rec["all"][k].append(v)
                rec[CATEGORY_NAMES[int(b["cat"][i])]][k].append(v)
            seen += 1
            if seen >= max_items:
                break
        if seen >= max_items:
            break
    summary = {g: {k: float(np.nanmean(v)) for k, v in d.items()} for g, d in rec.items()}
    summary["all"]["loss"] = float(np.mean(losses))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-root", required=True)
    ap.add_argument("--val-speech-root")
    ap.add_argument("--noise-root")
    ap.add_argument("--out-dir", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epoch-len", type=int, default=8000, help="virtual samples per epoch")
    ap.add_argument("--val-len", type=int, default=256)
    ap.add_argument("--eval-items", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out_dir, exist_ok=True)

    acfg, mcfg = AudioConfig(), ModelConfig()
    speech_tr = scan_files(args.speech_root)
    speech_va = scan_files(args.val_speech_root) if args.val_speech_root else speech_tr[-max(1, len(speech_tr) // 20):]
    noise = scan_noise_dir(args.noise_root) if args.noise_root else {}
    noise_tr, noise_va = split_noise(noise)

    train_ds = DefenceNoiseSpeechDataset(speech_tr, noise_tr, acfg, epoch_len=args.epoch_len)
    val_ds = DefenceNoiseSpeechDataset(speech_va, noise_va, acfg, epoch_len=args.val_len,
                                       deterministic=True, seed=1234)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=device.type == "cuda",
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, 8, shuffle=False, num_workers=min(2, args.workers))

    model = build_model(mcfg, acfg).to(device)
    stft = STFT(acfg.n_fft, acfg.hop).to(device)
    criterion = CompositeLoss(compress=acfg.compress).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(train_dl)

    def lr_lambda(step):                                          # linear warmup + cosine to 2 %
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_ep, best, step = 0, -1e9, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best, step = ck["epoch"] + 1, ck["best"], ck["step"]

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.out_dir)
    except Exception:
        writer = None
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M | device {device} | amp {amp} | "
          f"algorithmic latency {acfg.algorithmic_latency_ms:.0f} ms")

    for ep in range(start_ep, args.epochs):
        model.train()
        t0, run = time.time(), defaultdict(float)
        for it, b in enumerate(train_dl):
            noisy, clean = b["noisy"].to(device, non_blocking=True), b["clean"].to(device, non_blocking=True)
            with torch.no_grad():
                ref_spec = stft.analysis(clean)
            est_spec, est = run_model(model, stft, noisy, amp)
            loss, parts = criterion(est, clean, est_spec, ref_spec)    # losses in fp32

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1

            run["loss"] += loss.item()
            for k, v in parts.items():
                run[k] += v.item()
            if step % 50 == 0:
                n = it + 1
                print(f"ep {ep} it {it} loss {run['loss']/n:.4f} si-snr {run['si_snr']/n:.2f} dB "
                      f"gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e}")
                if writer:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("train/si_snr_db", parts["si_snr"].item(), step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0], step)

        val = evaluate(model, stft, criterion, val_dl, device, acfg.sr, args.eval_items)
        a = val["all"]
        print(f"== epoch {ep} ({time.time()-t0:.0f}s) val loss {a['loss']:.3f} | "
              f"SNR {a['snr_in']:.1f}->{a['snr_out']:.1f} dB (+{a['snr_gain']:.1f}) | "
              f"STOI {a['stoi_in']:.3f}->{a['stoi_out']:.3f} | PESQ {a['pesq_in']:.2f}->{a['pesq_out']:.2f}")
        for g, d in val.items():
            if g != "all":
                print(f"   {g:11s} dSNR {d['snr_gain']:5.1f}  STOI {d['stoi_out']:.3f}  PESQ {d['pesq_out']:.2f}")
        if writer:
            for g, d in val.items():
                for k, v in d.items():
                    writer.add_scalar(f"val/{g}/{k}", v, ep)

        score = a["pesq_out"] if np.isfinite(a["pesq_out"]) else -a["loss"]
        ck = checkpoint_dict(model, mcfg, acfg, opt=opt.state_dict(), sched=sched.state_dict(),
                             epoch=ep, step=step, best=max(best, score), val=val)
        torch.save(ck, os.path.join(args.out_dir, "last.pt"))
        if score > best:
            best = score
            torch.save(ck, os.path.join(args.out_dir, "best.pt"))
            print(f"   saved best.pt (score {score:.3f})")


if __name__ == "__main__":
    main()

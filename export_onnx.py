"""Module 5a: export ONE streaming step of the model to ONNX.

Graph:  (spec, c0, c1, c2, c3, h)  ->  (enh, c0_out, c1_out, c2_out, c3_out, h_out)
    spec   (1, 2, 129, frames)   `frames` is a DYNAMIC axis (1 for real-time, up to --max-frames
                                 for burst catch-up after a stall). Batch is fixed to 1.
    c*     conv caches (1, 2*cin, F, kt-1) and h GRU state (layers, 1, H): fixed shapes. The
           host feeds each *_out back as the next call's input.
FFT/iFFT are intentionally kept OUTSIDE the graph (TensorRT has no efficient FFT op; use
torch/cuFFT on the host, as in streaming.py).

    python -m anc.export_onnx --ckpt runs/crn_v1/best.pt --out crn_step.onnx --verify
"""
import argparse
import inspect

import numpy as np
import torch
import torch.nn as nn

from .model import load_model


class StreamStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, spec, *state):
        enh, new_state = self.m(spec, list(state))
        return (enh, *new_state)


def io_names(model):
    n = len(model.enc)
    ins = ["spec"] + [f"c{i}" for i in range(n)] + ["h"]
    outs = ["enh"] + [f"c{i}_out" for i in range(n)] + ["h_out"]
    return ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="crn_step.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    model, acfg = load_model(a.ckpt, "cpu")
    step = StreamStep(model).eval()
    state = model.init_state(1, "cpu")
    spec = torch.randn(1, 2, acfg.n_freq, 1)
    ins, outs = io_names(model)
    dyn = {"spec": {3: "frames"}, "enh": {3: "frames"}}

    extra = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        extra["dynamo"] = False                      # legacy tracer: stable names + dynamic_axes semantics
    torch.onnx.export(step, (spec, *state), a.out, input_names=ins, output_names=outs,
                      dynamic_axes=dyn, opset_version=a.opset, do_constant_folding=True, **extra)
    print(f"exported {a.out}")

    if a.verify:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        st_t = [s.clone() for s in state]
        st_o = [s.numpy() for s in state]
        worst = 0.0
        with torch.no_grad():
            for _ in range(24):                                      # 24 chained single-frame steps
                x = torch.randn(1, 2, acfg.n_freq, 1)
                y_t, st_t = model(x, st_t)
                res = sess.run(outs, {"spec": x.numpy(), **dict(zip(ins[1:], st_o))})
                st_o = res[1:]
                worst = max(worst, float(np.abs(res[0] - y_t.numpy()).max()))
            x = torch.randn(1, 2, acfg.n_freq, 5)                    # dynamic frames axis, fresh state
            y_t, _ = model(x, model.init_state(1, "cpu"))
            res = sess.run(outs, {"spec": x.numpy(), **{k: v.numpy() for k, v in zip(ins[1:], model.init_state(1, "cpu"))}})
            worst = max(worst, float(np.abs(res[0] - y_t.numpy()).max()))
        print(f"ONNX Runtime vs PyTorch max |diff| = {worst:.2e}  ({'OK' if worst < 1e-3 else 'MISMATCH'})")


if __name__ == "__main__":
    main()

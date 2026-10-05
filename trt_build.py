"""Module 5b: TensorRT FP16 / INT8 build and per-chunk latency benchmark (Jetson AGX Orin).

Run ON the Jetson (TensorRT python bindings ship with JetPack). For max clocks first:
    sudo nvpmodel -m 0 && sudo jetson_clocks

    python -m anc.trt_build build --onnx crn_step.onnx --engine crn_fp16.engine --precision fp16
    python -m anc.trt_build build --onnx crn_step.onnx --engine crn_int8.engine --precision int8 \
        --ckpt runs/crn_v1/best.pt --speech-root LibriSpeech/dev-clean --noise-root noise/
    python -m anc.trt_build bench --engine crn_fp16.engine

Notes
  * INT8: convolutions quantise well, the GRU generally stays FP16 (FP16 flag is set as fallback).
    ALWAYS re-run the STOI/PESQ evaluation on the INT8 engine and keep FP16 if quality drops.
  * Calibration uses real streaming states: the PyTorch model is run over noisy clips and every
    (frame, caches, h) input tuple is recorded, so activation ranges match deployment.
  * Equivalent CLI:  trtexec --onnx=crn_step.onnx --fp16 --saveEngine=crn_fp16.engine \
        --minShapes=spec:1x2x129x1 --optShapes=spec:1x2x129x1 --maxShapes=spec:1x2x129x8
"""
import argparse
import os
import time

import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError as e:  # allows importing the module off-device
    trt = None
    _TRT_ERR = e

TORCH_DT = {}
if trt is not None:
    TORCH_DT = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8}


# ------------------------------------------------------------------ INT8 calibration
def collect_records(ckpt, speech_root, noise_root, n_clips=6, n_records=400, device="cuda"):
    from .data import DefenceNoiseSpeechDataset, scan_files, scan_noise_dir
    from .model import load_model
    from .stft import STFT
    model, acfg = load_model(ckpt, device)
    stft = STFT(acfg.n_fft, acfg.hop).to(device)
    noise = scan_noise_dir(noise_root) if noise_root else {}
    ds = DefenceNoiseSpeechDataset(scan_files(speech_root), noise, acfg, epoch_len=n_clips,
                                   deterministic=True, seed=777)
    recs = []
    with torch.no_grad():
        for i in range(n_clips):
            spec = stft.analysis(ds[i]["noisy"].unsqueeze(0).to(device))
            state = model.init_state(1, device)
            for t in range(spec.shape[-1]):
                frame = spec[..., t:t + 1]
                recs.append([frame.cpu()] + [s.cpu() for s in state])
                _, state = model(frame, state)
    idx = np.linspace(0, len(recs) - 1, min(n_records, len(recs))).astype(int)
    return [recs[i] for i in idx]


if trt is not None:
    class StreamCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, records, input_names, cache_file):
            super().__init__()
            self.records, self.names, self.cache_file, self.i, self.bufs = records, input_names, cache_file, 0, None

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(self.records):
                return None
            rec = self.records[self.i]
            self.i += 1
            self.bufs = {n: t.contiguous().cuda() for n, t in zip(self.names, rec)}   # keep alive
            return [int(self.bufs[n].data_ptr()) for n in names]

        def read_calibration_cache(self):
            return open(self.cache_file, "rb").read() if os.path.exists(self.cache_file) else None

        def write_calibration_cache(self, cache):
            open(self.cache_file, "wb").write(cache)


# ------------------------------------------------------------------------- build
def build_engine(onnx_path, engine_path, precision="fp16", max_frames=8, workspace_gb=2, records=None):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):      # deprecated/implicit in TRT>=10
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    profile = builder.create_optimization_profile()
    names = []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        names.append(t.name)
        shp = list(t.shape)
        profile.set_shape(t.name,
                          tuple(1 if d == -1 else d for d in shp),            # min
                          tuple(1 if d == -1 else d for d in shp),            # opt  (real-time: 1 frame)
                          tuple(max_frames if d == -1 else d for d in shp))   # max
    config.add_optimization_profile(profile)

    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        assert records, "INT8 needs calibration records (--ckpt/--speech-root)"
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = StreamCalibrator(records, names, engine_path + ".calib")
        config.set_calibration_profile(profile)

    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"built {engine_path} ({precision}) in {time.time()-t0:.0f}s, {os.path.getsize(engine_path)/1e6:.1f} MB")


# --------------------------------------------------------------------- benchmark
def bench(engine_path, n_warm=200, n_iter=2000, hop_ms=8.0, n_fft=256, with_fft=False):
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    mode = engine.get_tensor_mode
    ins = [n for n in names if mode(n) == trt.TensorIOMode.INPUT]
    outs = [n for n in names if mode(n) == trt.TensorIOMode.OUTPUT]
    for n in ins:
        ctx.set_input_shape(n, tuple(1 if d == -1 else d for d in engine.get_tensor_shape(n)))
    bufs = {n: torch.zeros(tuple(ctx.get_tensor_shape(n)), dtype=TORCH_DT[engine.get_tensor_dtype(n)],
                           device="cuda") for n in ins + outs}
    for n, b in bufs.items():
        ctx.set_tensor_address(n, b.data_ptr())
    fb = {i: i + "_out" for i in ins if i != "spec"}                          # state feedback pairs
    win = torch.hann_window(n_fft, device="cuda").sqrt()
    frame = torch.randn(1, n_fft, device="cuda") * 0.05
    stream = torch.cuda.Stream()

    def step():
        if with_fft:
            Z = torch.fft.rfft(frame * win)
            bufs["spec"][0, 0, :, 0], bufs["spec"][0, 1, :, 0] = Z.real[0], Z.imag[0]
        ctx.execute_async_v3(stream.cuda_stream)
        for i, o in fb.items():
            bufs[i].copy_(bufs[o], non_blocking=True)
        if with_fft:
            E = torch.complex(bufs["enh"][:, 0, :, 0].float(), bufs["enh"][:, 1, :, 0].float())
            torch.fft.irfft(E, n=n_fft)

    times = []
    with torch.cuda.stream(stream):
        for k in range(n_warm + n_iter):
            t0 = time.perf_counter()
            step()
            stream.synchronize()                                              # per-chunk, as in a live loop
            if k >= n_warm:
                times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    tag = "engine + FFT/iFFT" if with_fft else "engine only"
    print(f"[{tag}] ms/chunk  mean {t.mean():.3f} p50 {np.percentile(t,50):.3f} p95 {np.percentile(t,95):.3f} "
          f"p99 {np.percentile(t,99):.3f} max {t.max():.3f} | budget {hop_ms:.1f} ms | RTF {t.mean()/hop_ms:.3f}")
    return t


def main():
    if trt is None:
        raise SystemExit(f"TensorRT not available: {_TRT_ERR}. Run this on the Jetson (JetPack).")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--onnx", required=True); b.add_argument("--engine", required=True)
    b.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    b.add_argument("--max-frames", type=int, default=8)
    b.add_argument("--ckpt"); b.add_argument("--speech-root"); b.add_argument("--noise-root")
    k = sub.add_parser("bench")
    k.add_argument("--engine", required=True)
    k.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()

    if a.cmd == "build":
        recs = collect_records(a.ckpt, a.speech_root, a.noise_root) if a.precision == "int8" else None
        build_engine(a.onnx, a.engine, a.precision, a.max_frames, records=recs)
    else:
        bench(a.engine, n_iter=a.iters, with_fft=False)
        bench(a.engine, n_iter=a.iters, with_fft=True)


if __name__ == "__main__":
    main()

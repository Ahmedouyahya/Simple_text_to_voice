"""VoxCPM synthesis worker — isolated subprocess so the parent can cancel us.

Reads one JSON config from stdin, synthesizes each chunk with the loaded
VoxCPM model, concatenates the parts with a short silence gap, and writes the
final wav to the requested path. Progress and diagnostic lines go to stderr so
the parent process can stream them live.

Input (stdin, single JSON object):
  {
    "chunks":               ["...", "..."],   required
    "out_wav":              "/abs/path.wav",  required
    "model_id":             "openbmb/VoxCPM-0.5B",  optional
    "device":               "auto",           optional: auto | cpu | cuda | mps
    "optimize":             null,             optional: null=auto, true, false
    "cfg_value":            2.0,              optional
    "inference_timesteps":  6,                optional
    "normalize":            false,            optional: run text normalization
    "max_len":              4096,             optional
    "silence_seconds":      0.35              optional
  }
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _resolve_device(requested: str) -> str:
    """Return the actual torch device string to use.

    'auto' picks the best available device.
    Requesting 'cpu' hides CUDA/MPS before torch is imported so the model
    never sees them (the cleanest way to force CPU in PyTorch).
    """
    requested = (requested or "auto").lower().strip()
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    # For auto/cuda/mps we let VoxCPM's own detection run, but we can report
    # what we expect based on availability after torch is imported.
    return requested  # 'auto', 'cuda', or 'mps'


def main() -> int:
    raw = sys.stdin.read()
    try:
        cfg = json.loads(raw)
    except Exception as e:
        log(f"FATAL: could not parse stdin JSON: {e}")
        return 2

    try:
        chunks  = cfg["chunks"]
        out_wav = cfg["out_wav"]
    except KeyError as e:
        log(f"FATAL: missing required key: {e}")
        return 2

    model_id        = cfg.get("model_id", "openbmb/VoxCPM-0.5B")
    device_req      = cfg.get("device", "auto")
    optimize_req    = cfg.get("optimize", None)   # None = auto
    cfg_value       = float(cfg.get("cfg_value", 2.0))
    timesteps       = int(cfg.get("inference_timesteps", 6))
    normalize       = bool(cfg.get("normalize", False))
    max_len         = int(cfg.get("max_len", 4096))
    silence_sec     = float(cfg.get("silence_seconds", 0.35))

    if not chunks:
        log("FATAL: no chunks to synthesize")
        return 2

    n = len(chunks)

    # Resolve device (must happen before importing torch so CUDA_VISIBLE_DEVICES
    # takes effect if the user wants CPU-only mode on a CUDA machine).
    _resolve_device(device_req)

    try:
        import torch
        import numpy as np
        import soundfile as sf
        from voxcpm import VoxCPM
    except Exception:
        log("FATAL: import failure")
        traceback.print_exc(file=sys.stderr)
        return 3

    # Determine the device VoxCPM will actually use (mirrors its own logic)
    if device_req == "cpu":
        detected = "cpu"
    elif torch.cuda.is_available():
        detected = "cuda"
    elif torch.backends.mps.is_available():
        detected = "mps"
    else:
        detected = "cpu"

    # Auto-decide whether to enable torch.compile:
    # - optimize=True only makes sense (and works) on CUDA
    # - on CPU/MPS it prints a warning and falls back anyway; skip it to be clean
    if optimize_req is None:
        optimize = (detected == "cuda")
    else:
        optimize = bool(optimize_req)

    log(f"voxcpm worker: model={model_id}, device={detected}, "
        f"optimize={optimize}, timesteps={timesteps}, cfg={cfg_value}, "
        f"normalize={normalize}, chunks={n}")

    log(f"loading {model_id} (first run downloads model weights from HuggingFace)")
    try:
        m = VoxCPM.from_pretrained(
            model_id,
            load_denoiser=False,
            optimize=optimize,
        )
    except Exception:
        log("FATAL: model load failed")
        traceback.print_exc(file=sys.stderr)
        return 4

    try:
        sr = int(m.tts_model.sample_rate)
    except Exception:
        sr = 16000
    log(f"model loaded — sample_rate={sr}, device={detected}")

    parts = []
    for i, chunk in enumerate(chunks):
        log(f"chunk {i + 1}/{n} ({len(chunk)} chars)")
        try:
            wav = m.generate(
                text=chunk,
                cfg_value=cfg_value,
                inference_timesteps=timesteps,
                normalize=normalize,
                max_len=max_len,
            )
        except Exception:
            log(f"FATAL: generate failed on chunk {i + 1}/{n}")
            traceback.print_exc(file=sys.stderr)
            return 5
        parts.append(wav)
        log(f"chunk {i + 1}/{n} done")

    log("concatenating parts")
    if len(parts) == 1:
        out = parts[0]
    else:
        gap = np.zeros(int(sr * silence_sec), dtype=parts[0].dtype)
        joined = []
        for i, p in enumerate(parts):
            joined.append(p)
            if i < len(parts) - 1:
                joined.append(gap)
        out = np.concatenate(joined)

    try:
        sf.write(out_wav, out, sr)
    except Exception:
        log(f"FATAL: writing {out_wav} failed")
        traceback.print_exc(file=sys.stderr)
        return 6

    log(f"wrote {out_wav} ({len(out)} samples @ {sr} Hz)")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

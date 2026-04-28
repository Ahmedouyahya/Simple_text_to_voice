"""VoxCPM synthesis worker — isolated subprocess so the parent can cancel us.

Reads one JSON config from stdin, synthesizes each chunk with the loaded
VoxCPM model, concatenates the parts with a short silence gap, and writes the
final wav to the requested path. Progress and diagnostic lines go to stderr so
the parent process can stream them live.

Input (stdin, single JSON object):
  {
    "chunks": ["...", "..."],
    "cfg_value": 2.0,
    "inference_timesteps": 6,
    "out_wav": "/abs/path/to/raw.wav",
    "silence_seconds": 0.35
  }
"""
from __future__ import annotations

import json
import sys
import traceback


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    raw = sys.stdin.read()
    try:
        cfg = json.loads(raw)
    except Exception as e:
        log(f"FATAL: could not parse stdin JSON: {e}")
        return 2

    try:
        chunks = cfg["chunks"]
        out_wav = cfg["out_wav"]
    except KeyError as e:
        log(f"FATAL: missing required key: {e}")
        return 2

    cfg_value   = float(cfg.get("cfg_value", 2.0))
    timesteps   = int(cfg.get("inference_timesteps", 6))
    silence_sec = float(cfg.get("silence_seconds", 0.35))

    if not chunks:
        log("FATAL: no chunks to synthesize")
        return 2

    n = len(chunks)
    log(f"voxcpm worker: {n} chunk(s), cfg={cfg_value}, timesteps={timesteps}")

    try:
        import numpy as np
        import soundfile as sf
        from voxcpm import VoxCPM
    except Exception:
        log("FATAL: import failure")
        traceback.print_exc(file=sys.stderr)
        return 3

    log("loading VoxCPM-0.5B (first run downloads ~1 GB)")
    try:
        m = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B", load_denoiser=False)
    except Exception:
        log("FATAL: model load failed")
        traceback.print_exc(file=sys.stderr)
        return 4
    try:
        sr = int(m.tts_model.sample_rate)
    except Exception:
        sr = 16000
    log(f"model loaded, sample_rate={sr}")

    parts = []
    for i, chunk in enumerate(chunks):
        log(f"chunk {i + 1}/{n} — {len(chunk)} chars")
        try:
            wav = m.generate(
                text=chunk,
                cfg_value=cfg_value,
                inference_timesteps=timesteps,
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

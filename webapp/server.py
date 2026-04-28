"""Piper Novel Studio — local web UI with auth.

FastAPI backend that orchestrates the Piper (or VoxCPM) → DeepFilterNet →
ffmpeg pipeline and streams progress over SSE to a single-page HTML frontend.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware

HERE = Path(__file__).resolve().parent.parent  # piper/
PYTHON = str(HERE / ".venv" / "bin" / "python")
DEEPFILTER = str(HERE / ".venv" / "bin" / "deepFilter")
VOICES_DIR = HERE / "voices"
OUTPUT_DIR = HERE / "output"
TOOLS = HERE / "tools"
STATIC = Path(__file__).parent / "static"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(TOOLS))
from clean_tex import clean as clean_latex  # noqa: E402

# Local auth module
sys.path.insert(0, str(Path(__file__).parent))
from auth import (  # noqa: E402
    change_credentials,
    seed_default,
    session_secret,
    verify,
)


ENGINES = [
    {"id": "piper",  "display": "Piper (fast, CPU-friendly)"},
    {"id": "voxcpm", "display": "VoxCPM-0.5B (slower, higher quality)"},
]

# Add any Piper voice here after downloading it with download_voices.sh.
# ID must match the filename in voices/ (without .onnx).
# Download any voice from: https://huggingface.co/rhasspy/piper-voices
VOICES_META = [
    # --- Arabic ---
    {"id": "ar_JO-kareem-medium", "lang": "ar", "engine": "piper",
     "display": "Kareem — Arabic (JO), medium"},

    # --- English ---
    {"id": "en_US-amy-medium",    "lang": "en", "engine": "piper",
     "display": "Amy — English (US), medium"},
    # {"id": "en_US-joe-medium",    "lang": "en", "engine": "piper",
    #  "display": "Joe — English (US), medium"},
    # {"id": "en_US-lessac-medium", "lang": "en", "engine": "piper",
    #  "display": "Lessac — English (US), medium"},
    # {"id": "en_GB-alan-medium",   "lang": "en", "engine": "piper",
    #  "display": "Alan — English (GB), medium"},

    # --- French ---
    # {"id": "fr_FR-upmc-medium",   "lang": "fr", "engine": "piper",
    #  "display": "UPMC — French (FR), medium"},

    # --- German ---
    # {"id": "de_DE-thorsten-medium", "lang": "de", "engine": "piper",
    #  "display": "Thorsten — German (DE), medium"},

    # --- Spanish ---
    # {"id": "es_ES-mls_10246-low", "lang": "es", "engine": "piper",
    #  "display": "MLS — Spanish (ES), low"},

    # --- Russian ---
    # {"id": "ru_RU-dmitri-medium", "lang": "ru", "engine": "piper",
    #  "display": "Dmitri — Russian (RU), medium"},

    # --- Chinese ---
    # {"id": "zh_CN-huayan-medium", "lang": "zh", "engine": "piper",
    #  "display": "Huayan — Chinese (CN), medium"},

    # --- VoxCPM (optional, slow on CPU) ---
    {"id": "voxcpm-default",      "lang": "multi", "engine": "voxcpm",
     "display": "VoxCPM default voice"},
]


def voice_path(voice_id: str) -> Path:
    return VOICES_DIR / f"{voice_id}.onnx"


# ---------- text cleanup ---------------------------------------------------

def clean_input(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data)
            pdf_path = f.name
        try:
            r = subprocess.run(
                ["pdftotext", "-nopgbrk", "-enc", "UTF-8", pdf_path, "-"],
                capture_output=True, text=True, check=True, timeout=60,
            )
            text = r.stdout
        finally:
            os.unlink(pdf_path)
        text = text.replace("\x0c", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    if ext == ".tex":
        src = data.decode("utf-8", errors="replace")
        cleaned = clean_latex(src)
        return cleaned
    return data.decode("utf-8", errors="replace").strip()


# ---------- VoxCPM chunking -----------------------------------------------

# VoxCPM has a hard 4096-token positional embedding limit. We chunk text
# conservatively so a chapter-sized paste never exceeds that.
VOXCPM_CHUNK_CHARS = 1200
VOXCPM_WORKER = TOOLS / "voxcpm_worker.py"


def _chunk_for_voxcpm(text: str, max_chars: int = VOXCPM_CHUNK_CHARS) -> list[str]:
    """Split `text` into VoxCPM-sized chunks on sentence boundaries.

    We prefer breaks at strong terminators (. ! ? ؟ …) and blank lines; a
    single sentence longer than `max_chars` falls back to word-level splitting
    so nothing ever exceeds the budget. Returns non-empty, stripped chunks.
    """
    text = text.strip()
    if not text:
        return []
    # Split into sentence-ish units. Include Arabic question mark ؟ and full stop ۔.
    sentences = re.split(r"(?<=[\.\!\?؟۔…])\s+|\n\s*\n", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if len(s) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            words = s.split()
            cur = ""
            for w in words:
                if len(cur) + len(w) + 1 > max_chars and cur:
                    chunks.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                chunks.append(cur)
            continue
        if len(buf) + len(s) + 1 > max_chars and buf:
            chunks.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks


# ---------- jobs -----------------------------------------------------------

@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    stage: str = "queued"
    error: str | None = None
    filename: str | None = None
    title: str = ""
    engine: str = ""
    voice: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    # Bounded history of every event we've emitted, for SSE reconnects.
    history: deque = field(default_factory=lambda: deque(maxlen=500))
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    task: asyncio.Task | None = None
    current_proc: asyncio.subprocess.Process | None = None
    cancelled: bool = False
    ended: bool = False


JOBS: dict[str, Job] = {}


def _safe_basename(title: str, job_id: str) -> str:
    base = re.sub(r"[^\w\s\u0600-\u06FF\-]+", "", title).strip()
    base = re.sub(r"\s+", "_", base)[:80]
    return base or f"job_{job_id[:8]}"


async def _emit(job: Job, **event):
    """Record the event in job history and fan it out to every live subscriber.

    Using per-subscriber queues (instead of a single queue) means a client that
    reconnects mid-job can replay `job.history` and then pick up live events
    without stealing them from the primary reader.

    NOTE: we only keep `log` and `status` events in history. Progress ticks
    come from the heartbeat at 1 Hz and would otherwise evict actual log lines
    from the bounded deque on long runs. Reconnecting clients get the current
    `job.progress` from the snapshot payload instead.
    """
    etype = event.get("type")
    if etype in ("log", "status"):
        job.history.append(event)
    if etype == "end":
        job.ended = True
    # Copy list to tolerate subscribers unsubscribing during iteration.
    for q in list(job.subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _job_summary(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "error": job.error,
        "filename": job.filename,
        "title": job.title,
        "engine": job.engine,
        "voice": job.voice,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
    }


async def _pump_stream(
    job: Job, stream: asyncio.StreamReader | None, tag: str, collected: list[str]
) -> None:
    """Read a subprocess stream line-by-line and emit each line as a log event."""
    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except Exception as e:
            await _emit(job, type="log", line=f"      [{tag} read error: {e}]", level="warn")
            return
        if not line:
            return
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text:
            continue
        collected.append(text)
        # Keep collected buffer from growing unbounded on chatty stderr.
        if len(collected) > 400:
            del collected[:200]
        await _emit(job, type="log", line=f"      {text}", level="subproc")


async def _heartbeat(job: Job, start_progress: float, end_progress: float,
                     estimated_seconds: float, label: str) -> None:
    """Interpolate progress between two checkpoints while a subprocess runs.

    Asymptotically approaches `end_progress` so we never overshoot if the task
    runs longer than estimated. Emits a tick every second so the UI stays
    responsive.
    """
    t0 = time.monotonic()
    while True:
        await asyncio.sleep(1.0)
        elapsed = time.monotonic() - t0
        span = max(end_progress - start_progress, 0.0)
        # 1 - e^(-elapsed/estimated): starts fast, eases out before end_progress.
        import math
        frac = 1.0 - math.exp(-elapsed / max(estimated_seconds, 1.0))
        pct = start_progress + span * frac * 0.97  # cap below end_progress
        if pct > job.progress:
            job.progress = pct
            await _emit(job, type="progress", progress=pct,
                        stage=label, elapsed=elapsed)


async def _run_subprocess_streamed(
    job: Job,
    cmd: list[str],
    *,
    stdin_bytes: bytes | None = None,
    start_progress: float,
    end_progress: float,
    estimated_seconds: float,
    label: str,
    fail_prefix: str,
) -> None:
    """Run a subprocess, stream its stderr/stdout live, and interpolate progress."""
    try:
        p = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"{fail_prefix}: cannot spawn `{cmd[0]}` — {e}") from e
    except OSError as e:
        raise RuntimeError(f"{fail_prefix}: spawn failed — {e}") from e
    job.current_proc = p
    collected_err: list[str] = []
    collected_out: list[str] = []

    async def _feed_stdin() -> None:
        if stdin_bytes is None or p.stdin is None:
            return
        try:
            p.stdin.write(stdin_bytes)
            await p.stdin.drain()
            p.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    heartbeat_task = asyncio.create_task(
        _heartbeat(job, start_progress, end_progress, estimated_seconds, label)
    )
    try:
        try:
            await asyncio.gather(
                _feed_stdin(),
                _pump_stream(job, p.stderr, "err", collected_err),
                _pump_stream(job, p.stdout, "out", collected_out),
                p.wait(),
            )
        except asyncio.CancelledError:
            # Parent task was cancelled (user hit Stop). Kill the subprocess
            # here so it doesn't outlive us, then propagate.
            if p.returncode is None:
                try:
                    p.kill()
                    await p.wait()
                except ProcessLookupError:
                    pass
            raise
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except BaseException:
            pass
        job.current_proc = None

    if p.returncode != 0:
        tail = "\n".join(collected_err[-30:]) or "\n".join(collected_out[-30:]) or "(no output)"
        raise RuntimeError(f"{fail_prefix} failed (exit {p.returncode}):\n{tail}")


async def _run_piper(job: Job, text: str, voice_id: str, raw: Path,
                     length_scale: float, sentence_silence: float) -> None:
    cmd = [
        PYTHON, "-m", "piper",
        "-m", str(voice_path(voice_id)),
        "-f", str(raw),
        "--length-scale", f"{length_scale}",
        "--sentence-silence", f"{sentence_silence}",
    ]
    # Piper on CPU runs ~2-5x realtime. Rough estimate: 1 char ≈ 50 ms of speech,
    # and synthesis ≈ 0.3x speech. Clamp so the heartbeat has something sane.
    est = max(8.0, min(len(text) * 0.02, 900.0))
    await _run_subprocess_streamed(
        job, cmd,
        stdin_bytes=text.encode("utf-8"),
        start_progress=0.03, end_progress=0.45,
        estimated_seconds=est,
        label="piper",
        fail_prefix="piper",
    )


async def _run_voxcpm(
    job: Job, text: str, raw: Path, cfg_value: float,
    inference_timesteps: int, phase_start: float, phase_end: float,
) -> None:
    """Run VoxCPM synthesis in a subprocess so Stop actually kills it."""
    chunks = _chunk_for_voxcpm(text)
    if not chunks:
        raise RuntimeError("VoxCPM: empty text after chunking")
    n = len(chunks)
    await _emit(job, type="log",
                line=f"      loading VoxCPM (timesteps={inference_timesteps}, chunks={n})")

    # Reality on a pure-CPU torch build: VoxCPM autoregressively decodes ~7
    # audio frames per character for Arabic, ~2.8s per frame (a fixed cost
    # dominated by the decoder pass, *not* by inference_timesteps). Plus
    # ~60s model load. Observed on this machine: 23 chars took 3m42s.
    total_chars = sum(len(c) for c in chunks)
    est = 60.0 + total_chars * 7.0 * 2.8
    est = max(60.0, min(est, 14400.0))  # cap at 4h

    payload = json.dumps({
        "chunks": chunks,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "out_wav": str(raw),
        "silence_seconds": 0.35,
    }).encode("utf-8")

    await _run_subprocess_streamed(
        job,
        [PYTHON, str(VOXCPM_WORKER)],
        stdin_bytes=payload,
        start_progress=phase_start,
        end_progress=phase_end,
        estimated_seconds=est,
        label=f"voxcpm ({n} chunk{'s' if n > 1 else ''})",
        fail_prefix="VoxCPM",
    )


async def run_synth(
    job: Job,
    text: str,
    engine: str,
    voice_id: str,
    title: str,
    length_scale: float = 1.0,
    sentence_silence: float = 0.25,
    cfg_value: float = 2.0,
    inference_timesteps: int = 6,
    skip_enhance: bool = False,
):
    try:
        job.status = "running"
        job.stage = "starting"
        await _emit(job, type="status", status="running", progress=0.0, stage="starting")

        base = _safe_basename(title, job.id)
        out_mp3 = OUTPUT_DIR / f"{base}.mp3"
        i = 2
        while out_mp3.exists():
            out_mp3 = OUTPUT_DIR / f"{base}_{i}.mp3"
            i += 1
        job.filename = out_mp3.name

        # Phase boundaries depend on whether we're running DeepFilterNet.
        # With enhance: synth 0.03→0.45, enhance 0.45→0.85, mp3 0.85→1.00
        # Without:     synth 0.03→0.85,                     mp3 0.85→1.00
        if skip_enhance:
            synth_end, enhance_end = 0.85, 0.85
            total_steps = 2
        else:
            synth_end, enhance_end = 0.45, 0.85
            total_steps = 3

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            raw = tmp / "raw.wav"

            # ---- Phase 1: synthesis -------------------------------------
            if engine == "voxcpm":
                job.stage = "voxcpm synthesis"
                await _emit(job, type="log",
                            line=f"[1/{total_steps}] VoxCPM synthesis "
                                 f"(timesteps={inference_timesteps}, cfg={cfg_value})")
                await _emit(job, type="progress", progress=0.03, stage=job.stage)
                await _run_voxcpm(
                    job, text, raw, cfg_value,
                    inference_timesteps=inference_timesteps,
                    phase_start=0.03, phase_end=synth_end,
                )
            else:
                job.stage = "piper synthesis"
                await _emit(
                    job, type="log",
                    line=f"[1/{total_steps}] Piper synthesis "
                         f"({voice_id}, speed×{length_scale}, silence {sentence_silence}s)",
                )
                await _emit(job, type="progress", progress=0.03, stage=job.stage)
                await _run_piper(job, text, voice_id, raw, length_scale, sentence_silence)

            if not raw.exists() or raw.stat().st_size == 0:
                raise RuntimeError("synthesis produced no audio (empty wav)")

            job.progress = synth_end
            await _emit(job, type="progress", progress=synth_end, stage="synthesis done")

            wav_bytes = raw.stat().st_size
            audio_seconds = wav_bytes / 44100.0

            # ---- Phase 2: DeepFilterNet (optional) ----------------------
            if skip_enhance:
                await _emit(job, type="log",
                            line="      skipping DeepFilterNet (clean-source mode)")
                dfn = raw
            else:
                job.stage = "deepfilter"
                await _emit(job, type="log",
                            line=f"[2/{total_steps}] DeepFilterNet enhancement")
                est_df = max(5.0, audio_seconds * 0.4)
                await _run_subprocess_streamed(
                    job, [DEEPFILTER, "-o", str(tmp), str(raw)],
                    start_progress=synth_end, end_progress=enhance_end,
                    estimated_seconds=est_df,
                    label="deepfilter",
                    fail_prefix="DeepFilterNet",
                )
                matches = sorted(tmp.glob("raw_DeepFilterNet*.wav"))
                dfn = matches[0] if matches else raw
                job.progress = enhance_end
                await _emit(job, type="progress", progress=enhance_end,
                            stage="enhance done")

            # ---- Phase 3: ffmpeg / loudness / mp3 -----------------------
            job.stage = "mp3 encode"
            step_num = 2 if skip_enhance else 3
            await _emit(job, type="log",
                        line=f"[{step_num}/{total_steps}] Loudness + MP3 encoding")
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(dfn),
                "-af", "highpass=f=60,acompressor=threshold=-18dB:ratio=3:attack=5:release=60,"
                       "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=22050",
                "-c:a", "libmp3lame", "-b:a", "96k", "-ac", "1",
                "-metadata", f"title={title}",
                "-metadata", f"artist={engine}:{voice_id} (TTS)",
                str(out_mp3),
            ]
            est_ff = max(3.0, audio_seconds * 0.08)
            await _run_subprocess_streamed(
                job, cmd,
                start_progress=enhance_end, end_progress=1.0,
                estimated_seconds=est_ff,
                label="ffmpeg",
                fail_prefix="ffmpeg",
            )

            job.progress = 1.0
            job.status = "done"
            job.stage = "complete"
            job.ended_at = time.time()
            await _emit(job, type="log", line=f"Done → {out_mp3.name}")
            await _emit(job, type="status", status="done", progress=1.0,
                        stage="complete", filename=out_mp3.name)

    except asyncio.CancelledError:
        job.status = "cancelled"
        job.stage = "cancelled"
        job.cancelled = True
        job.ended_at = time.time()
        if job.current_proc is not None and job.current_proc.returncode is None:
            try:
                job.current_proc.kill()
                await job.current_proc.wait()
            except ProcessLookupError:
                pass
        try:
            if job.filename:
                partial = OUTPUT_DIR / job.filename
                if partial.exists():
                    partial.unlink()
        except Exception:
            pass
        await _emit(job, type="log", line="Cancelled by user", level="warn")
        await _emit(job, type="status", status="cancelled", stage="cancelled")
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.stage = "error"
        job.ended_at = time.time()
        # Emit the full error as log lines too so it's visible in the log pane.
        for ln in str(e).splitlines() or [str(e)]:
            await _emit(job, type="log", line=ln, level="error")
        await _emit(job, type="status", status="error", stage="error", error=str(e))
    finally:
        await _emit(job, type="end")


# ---------- auth -----------------------------------------------------------

seed_default("137", "137")

app = FastAPI(title="Piper Novel Studio")
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret(),
    max_age=60 * 60 * 24 * 30,  # 30 days
    same_site="lax",
    https_only=False,
)


def require_user(request: Request) -> str:
    user = request.session.get("user")
    if not user:
        raise HTTPException(401, "login required")
    return user


class LoginIn(BaseModel):
    username: str
    password: str


class ChangeIn(BaseModel):
    current_password: str
    new_username: str = ""
    new_password: str = ""


@app.post("/api/login")
async def api_login(body: LoginIn, request: Request):
    if not verify(body.username, body.password):
        raise HTTPException(401, "invalid credentials")
    request.session["user"] = body.username
    return {"ok": True, "user": body.username}


@app.post("/api/logout")
async def api_logout(request: Request, _: str = Depends(require_user)):
    request.session.clear()
    return {"ok": True}


@app.get("/api/whoami")
async def api_whoami(request: Request):
    return {"user": request.session.get("user")}


@app.post("/api/change-credentials")
async def api_change(body: ChangeIn, request: Request, _: str = Depends(require_user)):
    if not body.new_username and not body.new_password:
        raise HTTPException(400, "nothing to change")
    ok = change_credentials(body.current_password, body.new_username, body.new_password)
    if not ok:
        raise HTTPException(400, "current password is wrong")
    if body.new_username:
        request.session["user"] = body.new_username
    return {"ok": True, "user": request.session.get("user")}


# ---------- routes ---------------------------------------------------------

@app.get("/")
async def index(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(STATIC / "index.html")


@app.get("/login")
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/", status_code=302)
    return FileResponse(STATIC / "login.html")


@app.get("/api/engines")
async def get_engines(_: str = Depends(require_user)):
    return ENGINES


@app.get("/api/voices")
async def get_voices(_: str = Depends(require_user)):
    return VOICES_META


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), _: str = Depends(require_user)):
    data = await file.read()
    try:
        text = clean_input(file.filename or "upload", data)
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, f"pdftotext failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(400, str(e))
    title = ""
    for line in text.splitlines():
        if line.strip():
            title = line.strip().rstrip(".")[:120]
            break
    return {"text": text, "title": title, "chars": len(text)}


class SynthIn(BaseModel):
    text: str
    engine: str = "piper"
    voice: str = "ar_JO-kareem-medium"
    title: str = "Untitled"
    length_scale: float = 1.0
    sentence_silence: float = 0.25
    cfg_value: float = 2.0
    # VoxCPM quality/speed knob. Fewer = faster with small quality drop.
    inference_timesteps: int = 6
    # Skip DeepFilterNet (VoxCPM output is usually clean already).
    skip_enhance: bool = False


@app.post("/api/synthesize")
async def api_synthesize(body: SynthIn, _: str = Depends(require_user)):
    if body.engine not in {e["id"] for e in ENGINES}:
        raise HTTPException(400, "unknown engine")
    if body.engine == "piper":
        if body.voice not in {v["id"] for v in VOICES_META if v["engine"] == "piper"}:
            raise HTTPException(400, "unknown voice")
        if not voice_path(body.voice).exists():
            raise HTTPException(500, f"voice model missing: {body.voice}")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    title = body.title.strip() or "Untitled"
    job = Job(
        id=uuid.uuid4().hex,
        title=title,
        engine=body.engine,
        voice=body.voice,
    )
    JOBS[job.id] = job
    # Clamp timesteps: VoxCPM becomes unusable below 3, diminishing returns above ~15.
    timesteps = max(3, min(int(body.inference_timesteps), 20))
    job.task = asyncio.create_task(run_synth(
        job, text,
        engine=body.engine,
        voice_id=body.voice,
        title=title,
        length_scale=body.length_scale,
        sentence_silence=body.sentence_silence,
        cfg_value=body.cfg_value,
        inference_timesteps=timesteps,
        skip_enhance=bool(body.skip_enhance),
    ))
    # Keep the job dict from growing without bound: trim oldest terminated jobs.
    terminated = [
        (jid, j) for jid, j in JOBS.items()
        if j.status in ("done", "error", "cancelled")
    ]
    if len(terminated) > 50:
        terminated.sort(key=lambda t: t[1].ended_at or 0)
        for jid, _ in terminated[:-50]:
            JOBS.pop(jid, None)
    return {"job_id": job.id}


@app.post("/api/cancel/{job_id}")
async def api_cancel(job_id: str, _: str = Depends(require_user)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job.status not in ("queued", "running"):
        return {"ok": False, "status": job.status}
    # Set the flag first so code running inside a thread-pool executor (e.g.
    # VoxCPM chunks) can check it between steps and bail out at a safe point.
    job.cancelled = True
    if job.task and not job.task.done():
        job.task.cancel()
    return {"ok": True}


@app.get("/api/jobs")
async def api_jobs(_: str = Depends(require_user)):
    """List recent jobs so the UI can recover state after a reload."""
    items = [_job_summary(j) for j in JOBS.values()]
    items.sort(key=lambda j: j["started_at"], reverse=True)
    return items


@app.get("/api/job/{job_id}")
async def api_job(job_id: str, _: str = Depends(require_user)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {
        **_job_summary(job),
        "history": list(job.history),
    }


@app.get("/api/progress/{job_id}")
async def api_progress(job_id: str, _: str = Depends(require_user)):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    job = JOBS[job_id]

    # Subscribe *before* snapshotting history so we don't drop events fired
    # between the snapshot and the subscription.
    q: asyncio.Queue = asyncio.Queue(maxsize=2000)
    job.subscribers.append(q)
    snapshot = list(job.history)
    already_ended = job.ended

    async def gen() -> AsyncIterator[dict]:
        try:
            # Send the full snapshot so a reconnecting client sees the log
            # and progress that happened before it arrived.
            yield {"data": json.dumps({
                "type": "snapshot",
                "job": _job_summary(job),
                "history": snapshot,
            })}
            if already_ended:
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "ping"})}
                    continue
                if ev.get("type") == "end":
                    yield {"data": json.dumps(ev)}
                    break
                yield {"data": json.dumps(ev)}
        finally:
            try:
                job.subscribers.remove(q)
            except ValueError:
                pass

    return EventSourceResponse(gen())


@app.get("/api/library")
async def api_library(_: str = Depends(require_user)):
    out = []
    for mp3 in sorted(OUTPUT_DIR.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True):
        dur = 0.0
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(r.stdout.strip() or 0)
        except Exception:
            pass
        out.append({
            "filename": mp3.name,
            "size": mp3.stat().st_size,
            "mtime": mp3.stat().st_mtime,
            "duration": dur,
        })
    return out


@app.get("/api/audio/{filename}")
async def api_audio(filename: str, _: str = Depends(require_user)):
    p = (OUTPUT_DIR / filename).resolve()
    if p.parent != OUTPUT_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/mpeg")


@app.delete("/api/library/{filename}")
async def api_delete(filename: str, _: str = Depends(require_user)):
    p = (OUTPUT_DIR / filename).resolve()
    if p.parent != OUTPUT_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    p.unlink()
    return {"ok": True}

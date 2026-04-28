# Simple Text to Voice

A local, offline text-to-speech web app built on [Piper TTS](https://github.com/rhasspy/piper). Paste or upload any text (`.txt`, `.tex`, `.pdf`), pick a voice, and get a clean MP3 — all in the browser, all on your machine, no internet required after setup.

## Features

- **Piper TTS** — fast, CPU-friendly neural TTS (Arabic & English voices pre-configured; any Piper voice works)
- **VoxCPM** — optional higher-quality engine (very slow on CPU — see warning below)
- **DeepFilterNet** — optional noise-removal pass before encoding
- **ffmpeg pipeline** — loudness normalization (EBU R128 −16 LUFS) → 96 kbps MP3
- **File upload** — drag-and-drop `.txt`, `.tex` (LaTeX auto-cleaned), or `.pdf`
- **Real-time progress** — streamed over SSE with auto-reconnect
- **Audio library** — in-browser playback, download, delete
- **Single-user auth** — scrypt-hashed password, changeable from the UI

## Quick start

### 1. Install system dependencies

```bash
# Debian / Ubuntu / Parrot OS
sudo apt install ffmpeg poppler-utils
```

### 2. Create a Python environment and install packages

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. Download voice models

Place the `.onnx` and `.onnx.json` files for each voice in the `voices/` directory.

| Voice | Language | Link |
|---|---|---|
| `ar_JO-kareem-medium` | Arabic (Jordan) | [Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/ar/ar_JO/kareem/medium) |
| `en_US-amy-medium` | English (US) | [Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium) |

Any other [Piper voice](https://huggingface.co/rhasspy/piper-voices) works — add its entry to `VOICES_META` in `webapp/server.py`.

### 4. Start the server

```bash
./serve.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — default login is `137` / `137` (change from the user menu).

## CLI tool

`read.sh` synthesizes a file directly from the terminal and produces four output variants for A/B comparison:

```bash
# Arabic
./read.sh ar my_text.txt

# English
./read.sh en my_text.txt output_name

# Pipe text directly (live playback via aplay)
echo "Hello world" | ./read.sh en -
```

Output files: `*_raw.wav`, `*_ff.wav` (ffmpeg chain), `*_dfn.wav` (DeepFilterNet), `*_both.wav` (DFN + ffmpeg).

## VoxCPM note

VoxCPM produces higher-quality output but is **~50× slower than realtime** on a CPU-only machine. Use Piper for anything beyond a short test sample.

## Project layout

```
.
├── webapp/
│   ├── server.py          # FastAPI backend: synthesis, auth, SSE, library
│   ├── auth.py            # scrypt password hashing + session secret
│   └── static/
│       ├── index.html     # main UI
│       └── login.html     # login page
├── tools/
│   ├── clean_tex.py       # generic LaTeX → plain text converter
│   └── voxcpm_worker.py   # VoxCPM subprocess worker
├── voices/                # put .onnx + .onnx.json files here (not tracked by git)
├── output/                # synthesized MP3s land here (not tracked by git)
├── read.sh                # CLI: synthesize one file
├── serve.sh               # start the web server
└── requirements.txt
```

## License

MIT

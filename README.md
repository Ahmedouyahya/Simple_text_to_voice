# Simple Text to Voice

A local, offline text-to-speech web app built on [Piper TTS](https://github.com/rhasspy/piper). Paste or upload any text (`.txt`, `.tex`, `.pdf`), pick a voice, and get a clean MP3 — all in the browser, all on your machine, no internet required after setup.

## Features

- **Piper TTS** — fast, CPU-friendly neural TTS (supports 30+ languages)
- **VoxCPM** — optional higher-quality engine (very slow on CPU — see note below)
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

Run the included script to download the default voices automatically:

```bash
./download_voices.sh
```

This downloads `ar_JO-kareem-medium` (Arabic) and `en_US-amy-medium` (English) from Hugging Face into `voices/`.

To download a different or additional voice:

```bash
./download_voices.sh en_US-joe-medium
./download_voices.sh de_DE-thorsten-medium fr_FR-upmc-medium
```

### 4. Start the server

```bash
./serve.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1.8000) — default login is `137` / `137` (change from the user menu).

---

## Available voices

Piper supports 30+ languages. Below are recommended voices per language. Download any of them with `./download_voices.sh <voice-id>`, then uncomment the matching entry in `VOICES_META` inside `webapp/server.py`.

All voices are hosted at: `https://huggingface.co/rhasspy/piper-voices`

| Voice ID | Language | Gender | Quality |
|---|---|---|---|
| `ar_JO-kareem-medium` | Arabic | Male | medium |
| `en_US-amy-medium` | English (US) | Female | medium |
| `en_US-joe-medium` | English (US) | Male | medium |
| `en_US-lessac-medium` | English (US) | Female | medium |
| `en_US-libritts_r-medium` | English (US) | Neutral | medium |
| `en_GB-alan-medium` | English (UK) | Male | medium |
| `en_GB-jenny_dioco-medium` | English (UK) | Female | medium |
| `fr_FR-upmc-medium` | French | Male | medium |
| `fr_FR-mls-medium` | French | Neutral | medium |
| `de_DE-thorsten-medium` | German | Male | medium |
| `de_DE-eva_k-x_low` | German | Female | x_low |
| `es_ES-mls_10246-low` | Spanish (ES) | Neutral | low |
| `es_MX-claude-high` | Spanish (MX) | Male | high |
| `pt_BR-faber-medium` | Portuguese (BR) | Male | medium |
| `ru_RU-dmitri-medium` | Russian | Male | medium |
| `ru_RU-irina-medium` | Russian | Female | medium |
| `zh_CN-huayan-medium` | Chinese | Female | medium |
| `tr_TR-dfki-medium` | Turkish | Neutral | medium |
| `ja_JP-kenichi-medium` | Japanese | Male | medium |
| `it_IT-riccardo-x_low` | Italian | Male | x_low |

> Full list: [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

### Adding a voice to the UI

1. Download the model:
   ```bash
   ./download_voices.sh en_US-joe-medium
   ```
2. Open `webapp/server.py` and uncomment (or add) an entry in `VOICES_META`:
   ```python
   {"id": "en_US-joe-medium", "lang": "en", "engine": "piper",
    "display": "Joe — English (US), medium"},
   ```
3. Restart the server — the new voice appears in the dropdown immediately.

---

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

---

## VoxCPM note

VoxCPM produces higher-quality output but is **~50× slower than realtime** on a CPU-only machine. Use Piper for anything beyond a short test sample.

---

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
├── voices/                # .onnx + .onnx.json files go here (not tracked by git)
├── output/                # synthesized MP3s land here (not tracked by git)
├── download_voices.sh     # download voice models from Hugging Face
├── read.sh                # CLI: synthesize one file
├── serve.sh               # start the web server
└── requirements.txt
```

## License

MIT

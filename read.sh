#!/usr/bin/env bash
# Usage:
#   ./read.sh ar input.txt              -> output/input_{raw,ff,dfn,both}.wav
#   ./read.sh en input.txt              -> output/input_{raw,ff,dfn,both}.wav
#   ./read.sh ar input.txt myname       -> output/myname_{raw,ff,dfn,both}.wav
#   echo "نص" | ./read.sh ar -          -> play raw directly (no cleanup)
#
# Every run produces four files so you can A/B the post-processing:
#   *_raw.wav   Piper output, untouched
#   *_ff.wav    ffmpeg chain (highpass + compressor + loudnorm)
#   *_dfn.wav   DeepFilterNet neural cleanup
#   *_both.wav  DeepFilterNet -> ffmpeg chain

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HERE/.venv/bin/python"
DEEPFILTER="$HERE/.venv/bin/deepFilter"
VOICES="$HERE/voices"
OUT="$HERE/output"
mkdir -p "$OUT"

LANG="${1:-}"
INPUT="${2:-}"
BASENAME="${3:-}"

case "$LANG" in
  ar) MODEL="$VOICES/ar_JO-kareem-medium.onnx" ;;
  en) MODEL="$VOICES/en_US-amy-medium.onnx" ;;
  *)  echo "Usage: $0 {ar|en} <input.txt|-> [basename]"; exit 1 ;;
esac

if [ -z "$INPUT" ]; then
  echo "Usage: $0 {ar|en} <input.txt|-> [basename]"; exit 1
fi

# Live-playback mode: stream to aplay, no post-processing
if [ "$INPUT" = "-" ]; then
  "$PYTHON" -m piper -m "$MODEL" --output-raw | aplay -r 22050 -f S16_LE -t raw -
  exit 0
fi

if [ ! -f "$INPUT" ]; then
  echo "File not found: $INPUT"; exit 1
fi

[ -z "$BASENAME" ] && BASENAME="$(basename "${INPUT%.*}")"

RAW="$OUT/${BASENAME}_raw.wav"
FF="$OUT/${BASENAME}_ff.wav"
DFN="$OUT/${BASENAME}_dfn.wav"
BOTH="$OUT/${BASENAME}_both.wav"

echo "[1/4] Synthesizing raw      -> $RAW"
"$PYTHON" -m piper -m "$MODEL" -f "$RAW" < "$INPUT"

echo "[2/4] ffmpeg chain          -> $FF"
ffmpeg -y -loglevel error -i "$RAW" \
  -af "highpass=f=60,acompressor=threshold=-18dB:ratio=3:attack=5:release=60,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=22050" \
  -c:a pcm_s16le "$FF"

if [ -x "$DEEPFILTER" ]; then
  echo "[3/4] DeepFilterNet         -> $DFN"
  TMPDIR="$(mktemp -d)"
  "$DEEPFILTER" -o "$TMPDIR" "$RAW" >/dev/null 2>&1
  mv "$TMPDIR"/*.wav "$DFN"
  rmdir "$TMPDIR"

  echo "[4/4] DFN + ffmpeg chain    -> $BOTH"
  ffmpeg -y -loglevel error -i "$DFN" \
    -af "highpass=f=60,acompressor=threshold=-18dB:ratio=3:attack=5:release=60,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=22050" \
    -c:a pcm_s16le "$BOTH"
else
  echo "[3/4] DeepFilterNet not installed — skipping _dfn and _both"
fi

echo
echo "Done. Compare:"
ls -lh "$OUT"/"${BASENAME}"_*.wav 2>/dev/null

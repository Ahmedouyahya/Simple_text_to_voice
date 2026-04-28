#!/usr/bin/env bash
# Download Piper voice models from Hugging Face into voices/
#
# Usage:
#   ./download_voices.sh                       # download the two default voices
#   ./download_voices.sh en_US-joe-medium      # one specific voice
#   ./download_voices.sh en_US-amy-medium fr_FR-upmc-medium   # multiple

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICES="$HERE/voices"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Voice ID format: {lang_code}-{dataset}-{quality}
# Examples: ar_JO-kareem-medium  en_US-amy-medium  de_DE-thorsten-medium
download_voice() {
  local VOICE_ID="$1"

  # Parse voice ID into parts
  local LANG_CODE="${VOICE_ID%%-*}"           # ar_JO
  local REST="${VOICE_ID#*-}"                 # kareem-medium
  local DATASET="${REST%-*}"                  # kareem
  local QUALITY="${REST##*-}"                 # medium
  local LANG_FAMILY="${LANG_CODE%%_*}"        # ar

  local DIR="$BASE_URL/$LANG_FAMILY/$LANG_CODE/$DATASET/$QUALITY"

  for EXT in onnx.json onnx; do
    local DEST="$VOICES/$VOICE_ID.$EXT"
    if [ -f "$DEST" ]; then
      echo "  already have: $VOICE_ID.$EXT"
      continue
    fi
    echo "  downloading $VOICE_ID.$EXT ..."
    if command -v wget &>/dev/null; then
      wget -q --show-progress -O "$DEST" "$DIR/$VOICE_ID.$EXT"
    elif command -v curl &>/dev/null; then
      curl -L --progress-bar -o "$DEST" "$DIR/$VOICE_ID.$EXT"
    else
      echo "Error: neither wget nor curl found. Install one and retry." >&2
      return 1
    fi
  done
}

mkdir -p "$VOICES"

if [ "$#" -eq 0 ]; then
  # Default voices — matches the pre-configured entries in webapp/server.py
  VOICES_LIST=(ar_JO-kareem-medium en_US-amy-medium)
else
  VOICES_LIST=("$@")
fi

for V in "${VOICES_LIST[@]}"; do
  echo "[ $V ]"
  download_voice "$V"
  echo
done

echo "Done. Models in $VOICES:"
ls "$VOICES"/*.onnx 2>/dev/null | xargs -I{} basename {} || echo "  (none yet)"

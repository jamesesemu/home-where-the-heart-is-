#!/usr/bin/env bash
# Full rebuild: raw VO -> timing-locked mix -> captions -> 2160x3840 video.
# Usage: ./render.sh <raw_vo.mp3> <whisper_model_dir> <out_dir>
set -euo pipefail
RAW=$1; MODEL=$2; OUT=$3; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
ffmpeg -hide_banner -loglevel error -y -i "$RAW" -ar 48000 -ac 1 -c:a pcm_s16le "$OUT/raw.wav"
python3 "$HERE/build_audio.py" "$OUT/raw.wav" "$OUT"

# Master: two-pass linear loudnorm to -14 LUFS / -1 dBTP (no dynamic pumping)
J=$(ffmpeg -hide_banner -i "$OUT/mix_premaster.wav" -af loudnorm=I=-14:TP=-1:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
g() { echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -hide_banner -loglevel error -y -i "$OUT/mix_premaster.wav" \
  -af "loudnorm=I=-14:TP=-1:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true,aresample=48000" \
  -c:a pcm_s24le "$OUT/final_mix.wav"

# Captions
ffmpeg -hide_banner -loglevel error -y -i "$OUT/stem_dialogue.wav" -ac 1 -ar 16000 -f f32le "$OUT/dlg16.f32"
python3 "$HERE/align_words.py" "$OUT/cues.json" "$OUT/dlg16.f32" "$MODEL" "$OUT/word_times.json"
python3 "$HERE/build_subs.py" "$OUT/cues.json" "$OUT/subtitles.ass" "$OUT/word_times.json"

# Video: solid black 2160x3840, subtitles burned in, final mix
DUR=$(python3 -c "import json;print(json.load(open('$OUT/cues.json'))['duration'])")
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "color=c=black:s=2160x3840:r=30:d=$DUR" -i "$OUT/final_mix.wav" \
  -vf "subtitles=$OUT/subtitles.ass:fontsdir=$HERE/fonts" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -profile:v high -level 5.2 \
  -c:a aac -b:a 320k -ar 48000 -shortest -movflags +faststart "$OUT/Short15_final_2160x3840.mp4"
rm -f "$OUT/dlg16.f32"
echo "done -> $OUT/Short15_final_2160x3840.mp4"

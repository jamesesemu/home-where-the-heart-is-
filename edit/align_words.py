"""Word-level timings for each dialogue cue (faster-whisper), used to place
caption breaks exactly on word boundaries.

Usage: python3 align_words.py <cues.json> <dialogue_16k_mono.f32> <whisper_model_dir> <out.json>
Cues whose recognised word count differs from the script fall back to
proportional timing inside build_subs.py.
"""
import json
import sys

import numpy as np
from faster_whisper import WhisperModel

cues = json.load(open(sys.argv[1]))["cues"]
a = np.fromfile(sys.argv[2], dtype=np.float32)
sr = 16000
m = WhisperModel(sys.argv[3], device="cpu", compute_type="int8")
out = {}
for i, c in enumerate(cues):
    text = " ".join(w for w in c["text"].split() if w != "/")
    s0 = c["start"] - 0.1
    seg = a[int(s0 * sr): int((c["end"] + 0.15) * sr)]
    segs, _ = m.transcribe(seg, language="en", word_timestamps=True, initial_prompt=text)
    ws = [w for sg in segs for w in sg.words]
    ok = len(ws) == len(text.split())
    print(f"{i:2d} {'OK ' if ok else 'n/a'} {text} | {' '.join(w.word.strip() for w in ws)}")
    if ok:
        out[str(i)] = [[round(s0 + w.start, 3), round(s0 + w.end, 3)] for w in ws]
json.dump(out, open(sys.argv[4], "w"))

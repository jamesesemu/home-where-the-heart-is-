"""Builds the timing-locked audio mix for "Short 15" (FaceTime) from the raw VO.

Usage: python3 build_audio.py <raw_vo.wav (48k mono)> <out_dir>

Everything is placed on one absolute timeline (seconds), so the animator can
work straight from the exported mix. Dialogue placements are also written to
cues.json, which build_subs.py turns into the subtitle file.
"""
import json
import sys

import numpy as np

SR = 48000
RNG = np.random.default_rng(15)


# ----------------------------------------------------------------- helpers
def secs(n):
    return int(round(n * SR))


def fade(x, fin=0.01, fout=0.02):
    x = x.copy()
    a, b = min(len(x), secs(fin)), min(len(x), secs(fout))
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def env(n, a, d, sus=0.0, r=0.0, sl=0.0):
    """Simple ADSR (seconds) over n samples."""
    e = np.zeros(n)
    ia, id_ = secs(a), secs(d)
    ia = max(ia, 1)
    e[:ia] = np.linspace(0, 1, ia)[: n]
    end = min(n, ia + id_)
    if end > ia:
        e[ia:end] = np.linspace(1, sl, end - ia)
    e[end:] = sl
    if r:
        ir = min(n, secs(r))
        e[-ir:] *= np.linspace(1, 0, ir)
    return e


def lp(x, cutoff):
    """One-pole low-pass."""
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def lp_fast(x, cutoff):
    # FFT brick-ish low-pass with soft knee: fast for long buffers
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def hp_fast(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (cutoff / np.maximum(f, 1e-3)) ** 4)
    return np.fft.irfft(X, len(x))


def bp_fast(x, lo, hi):
    return hp_fast(lp_fast(x, hi), lo)


def noise(dur):
    return RNG.standard_normal(secs(dur))


def tone(freq, dur, kind="sine"):
    t = np.arange(secs(dur)) / SR
    if kind == "sine":
        return np.sin(2 * np.pi * freq * t)
    if kind == "tri":
        return 2 / np.pi * np.arcsin(np.sin(2 * np.pi * freq * t))
    raise ValueError(kind)


def pluck(freq, dur, bright=0.5):
    """Karplus-Strong plucked string (guitar/uke-ish)."""
    n = secs(dur)
    p = max(2, int(SR / freq))
    buf = RNG.uniform(-1, 1, p)
    buf = lp_fast(np.tile(buf, 4), 2000 + 6000 * bright)[:p]
    out = np.empty(n)
    decay = 0.996
    for i in range(n):
        v = buf[i % p]
        out[i] = v
        buf[i % p] = decay * 0.5 * (v + buf[(i + 1) % p])
    return out * env(n, 0.002, dur, sl=1.0, r=0.05)


def place(out, s, i, gain=1.0):
    """Add s into out at sample i, clipped to out's bounds."""
    if i >= len(out):
        return
    j = min(len(out), i + len(s))
    out[i:j] += s[: j - i] * gain


def db(v):
    return 10 ** (v / 20)


def norm_rms(x, target_db):
    r = np.sqrt(np.mean(x ** 2)) + 1e-12
    return x * db(target_db) / r


class Track:
    def __init__(self, dur):
        self.L = np.zeros(secs(dur))
        self.R = np.zeros(secs(dur))

    def put(self, st, at, gain_db=0.0):
        """Add a stereo (n, 2) clip at time `at`."""
        i = secs(at)
        if i >= len(self.L):
            return
        j = min(len(self.L), i + len(st))
        g = db(gain_db)
        self.L[i:j] += st[: j - i, 0] * g
        self.R[i:j] += st[: j - i, 1] * g

    def add(self, x, at, gain_db=0.0, pan=0.0):
        i = secs(at)
        x = x * db(gain_db)
        j = min(len(self.L), i + len(x))
        x = x[: j - i]
        gl, gr = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
        self.L[i:j] += x * gl * np.sqrt(2)
        self.R[i:j] += x * gr * np.sqrt(2)


# ------------------------------------------------------------------ SFX
def sfx_calling_loop(dur):
    """Soft, generic video-call 'connecting' ripple (original synth, not a
    copy of any real ringtone): repeating rising two-note marimba blips."""
    out = np.zeros(secs(dur))
    t = 0.0
    while t < dur - 0.05:
        for k, f in enumerate([659.25, 880.0, 987.77]):
            n = secs(0.35)
            s = tone(f, 0.35) + 0.3 * tone(f * 4, 0.35)
            s *= env(n, 0.004, 0.33, sl=0.0)
            place(out, s, secs(t + k * 0.11), 0.6)
        t += 1.0
    return fade(out, 0.005, 0.08)


def sfx_connect():
    """Bright upward 'call connected' chime."""
    out = np.zeros(secs(0.9))
    for k, f in enumerate([783.99, 1046.5, 1567.98]):
        n = secs(0.6)
        s = (tone(f, 0.6) + 0.25 * tone(2 * f, 0.6)) * env(n, 0.003, 0.58, sl=0.0)
        place(out, s * 0.55, secs(k * 0.07))
    return out


def sfx_end():
    """Descending 'call ended' boop."""
    out = np.zeros(secs(0.6))
    for k, f in enumerate([880.0, 587.33]):
        n = secs(0.32)
        s = (tone(f, 0.32) + 0.2 * tone(3 * f, 0.32)) * env(n, 0.003, 0.3, sl=0.0)
        place(out, s * 0.6, secs(k * 0.14))
    return out


def sfx_whoosh(dur=0.45):
    n = secs(dur)
    x = noise(dur)
    # sweep a band upward then fade
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    lo = bp_fast(x, 300, 1500)
    hi = bp_fast(x, 1500, 6000)
    mix = np.linspace(0, 1, n)
    return (lo * (1 - mix) + hi * mix) * e * 0.5


def sfx_thud(weight=1.0):
    """Phone smacking a wooden table: low body + click."""
    n = secs(0.35)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (140 * weight ** -0.3) * t * (1 - 0.4 * t)) * np.exp(-t * 28)
    knock = bp_fast(noise(0.35), 800, 5000) * np.exp(-t * 90)
    return (body * 0.9 + knock * 0.7) * weight


def sfx_rattle(dur):
    """Phone clattering/sliding: dense small plastic/glass ticks."""
    out = np.zeros(secs(dur))
    t = 0.0
    while t < dur - 0.03:
        n = secs(0.03)
        tick = bp_fast(noise(0.03), 2500, 9000) * np.exp(-np.arange(n) / SR * 260)
        i = secs(t)
        place(out, tick * RNG.uniform(0.2, 0.7), i)
        t += RNG.uniform(0.012, 0.05)
    return out


def sfx_fumble(dur):
    """Fabric/hand fumble rustle."""
    x = bp_fast(noise(dur), 1200, 7000)
    e = np.abs(lp_fast(RNG.standard_normal(secs(dur)), 8)) * 3
    return x * np.clip(e, 0, 1) * 0.35


def sfx_hair(dur):
    """Fast hair-ruffle swishes while Johnny fixes his hair."""
    out = np.zeros(secs(dur))
    for k in range(int(dur / 0.28)):
        s = sfx_fumble(0.22) * np.hanning(secs(0.22))
        i = secs(k * 0.28 + RNG.uniform(0, 0.05))
        place(out, s, i)
    return out * 0.8


def sfx_crunch():
    """Single peanut crunch / chew."""
    d = RNG.uniform(0.12, 0.2)
    n = secs(d)
    x = bp_fast(noise(d), 1500, 9000)
    grains = (RNG.uniform(0, 1, n) > 0.85).astype(float)
    grains = lp_fast(grains, 900)
    return x * np.clip(grains * 4, 0, 1) * np.hanning(n) * 0.9


def sfx_chewing(dur, rate=2.4):
    out = np.zeros(secs(dur))
    t = 0.0
    while t < dur - 0.25:
        c = sfx_crunch()
        i = secs(t)
        place(out, c, i)
        t += 1 / rate * RNG.uniform(0.8, 1.2)
    return out


def sfx_footsteps(dur, pace=0.52):
    out = np.zeros(secs(dur))
    t = 0.0
    while t < dur - 0.2:
        n = secs(0.12)
        tt = np.arange(n) / SR
        step = bp_fast(noise(0.12), 120, 2500) * np.exp(-tt * 45)
        step += 0.6 * np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 40)
        i = secs(t)
        place(out, step * RNG.uniform(0.6, 0.9), i)
        t += pace * RNG.uniform(0.95, 1.05)
    return out


def amb_street(dur):
    """Outdoor street bed: low rumble + distant traffic swells + birds."""
    n = secs(dur)
    rumble = lp_fast(noise(dur), 250) * 0.6
    traffic = bp_fast(noise(dur), 200, 1800)
    swell = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / SR / 6.5 + 1.0))
    out = rumble + traffic * swell * 0.25
    # a couple of synth bird chirps
    for at in RNG.uniform(0.5, dur - 0.5, int(dur / 2.5)):
        cn = secs(0.09)
        tt = np.arange(cn) / SR
        f = 3800 + 1400 * np.sin(np.pi * tt / 0.09)
        ch = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(cn) * 0.25
        for r in range(RNG.integers(1, 3)):
            i = secs(at + r * 0.13)
            place(out, ch, i)
    return fade(out, 0.3, 0.3)


def amb_tv(dur):
    """Muffled TV across the room: band-limited babble-like modulated noise."""
    n = secs(dur)
    x = bp_fast(noise(dur), 250, 2200)
    syll = np.clip(lp_fast(RNG.standard_normal(n), 5) * 6, 0, 1)
    return fade(x * (0.3 + syll), 0.3, 0.3) * 0.5


def amb_room(dur):
    return lp_fast(noise(dur), 600) * 0.4


# ---------------------------------------------------------------- music
NOTE = {n: 440 * 2 ** ((i - 9) / 12) for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def f_of(name, octave):
    return NOTE[name] * 2 ** (octave - 4)


def music_bouncy(dur, bpm=118):
    """Light, sunny ukulele-style strum loop (C - Am - F - G)."""
    beat = 60 / bpm
    prog = [("C", ["C", "E", "G"]), ("A", ["A", "C", "E"]),
            ("F", ["F", "A", "C"]), ("G", ["G", "B", "D"])]
    out = np.zeros(secs(dur) + secs(2))
    t, k = 0.0, 0
    pattern = [0, 1, 1.5, 2.5, 3]  # strum positions within a bar (beats)
    while t < dur:
        root, chord = prog[k % 4]
        for pos in pattern:
            st = t + pos * beat
            for j, nn in enumerate(chord):
                s = pluck(f_of(nn, 4 if nn >= root else 5), 0.6, 0.6)
                i = secs(st + j * 0.012)
                place(out, s * 0.22, i)
        # bass on 1 and 3
        for pos in (0, 2):
            b = tone(f_of(root, 2), beat * 1.6, "tri") * env(secs(beat * 1.6), 0.01, beat * 1.5, sl=0.0)
            i = secs(t + pos * beat)
            place(out, b * 0.35, i)
        # light shaker
        for e8 in range(8):
            sh = hp_fast(noise(0.05), 5000) * np.exp(-np.arange(secs(0.05)) / SR * 80)
            i = secs(t + e8 * beat / 2)
            place(out, sh * (0.08 if e8 % 2 else 0.05), i)
        t += 4 * beat
        k += 1
    return out[: secs(dur)]


def music_awkward(dur, bpm=92):
    """Soft, slightly nervous plucked arpeggio (Fmaj7 - Em7)."""
    beat = 60 / bpm
    chords = [["F", "A", "C", "E"], ["E", "G", "B", "D"]]
    out = np.zeros(secs(dur) + secs(2))
    t, k = 0.0, 0
    while t < dur:
        ch = chords[k % 2]
        seq = ch + ch[::-1][1:3]
        for j, nn in enumerate(seq):
            s = pluck(f_of(nn, 4 if j < 4 else 5), 0.9, 0.35)
            i = secs(t + j * beat / 2 * 1.333)
            place(out, s * 0.2, i)
        b = tone(f_of(ch[0], 2), beat * 3.8, "sine") * env(secs(beat * 3.8), 0.05, beat * 3.7, sl=0.0)
        place(out, b * 0.3, secs(t))
        t += 4 * beat
        k += 1
    return out[: secs(dur)]


def music_chill(dur, bpm=100):
    """Laid-back walking groove (Dm7 - G7 - Cmaj7 - Am7) for the street call."""
    beat = 60 / bpm
    prog = [("D", ["F", "A", "C"]), ("G", ["F", "B", "D"]),
            ("C", ["E", "G", "B"]), ("A", ["G", "C", "E"])]
    out = np.zeros(secs(dur) + secs(2))
    t, k = 0.0, 0
    while t < dur:
        root, chord = prog[k % 4]
        for pos in (0.5, 1.5, 2.5, 3.5):  # off-beat skank
            for j, nn in enumerate(chord):
                s = pluck(f_of(nn, 4), 0.35, 0.4)
                i = secs(t + pos * beat + j * 0.008)
                place(out, s * 0.17, i)
        for pos in range(4):
            bn = root if pos % 2 == 0 else chord[1]
            b = tone(f_of(bn, 2), beat * 0.9, "tri") * env(secs(beat * 0.9), 0.005, beat * 0.85, sl=0.0)
            i = secs(t + pos * beat)
            place(out, b * 0.32, i)
        for e8 in range(4):
            kk = lp_fast(noise(0.12), 150) * np.exp(-np.arange(secs(0.12)) / SR * 35)
            i = secs(t + e8 * beat)
            place(out, kk * (0.5 if e8 % 2 == 0 else 0.0), i)
            hh = hp_fast(noise(0.04), 6000) * np.exp(-np.arange(secs(0.04)) / SR * 120)
            i = secs(t + e8 * beat + beat / 2)
            place(out, hh * 0.1, i)
        t += 4 * beat
        k += 1
    return out[: secs(dur)]



# ------------------------------------------------------------ samples
import os
import subprocess

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_CACHE = {}


def smp(name, start=0.0, dur=None, fin=0.005, fout=0.03):
    """Load a recorded sound from assets/ as stereo float (n, 2), trimmed."""
    if name not in _CACHE:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(ASSETS, name),
                              "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        _CACHE[name] = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)
    x = _CACHE[name][secs(start): secs(start + dur) if dur else None].copy()
    for ch in range(2):
        x[:, ch] = fade(x[:, ch], fin, fout)
    return x


def rms_to(x, target_db):
    """RMS-normalise a stereo clip (active part only) to target dBFS."""
    m = np.abs(x).mean(1)
    act = x[m > m.max() * 0.05] if m.max() > 0 else x
    r = np.sqrt(np.mean(act ** 2)) + 1e-12
    return x * db(target_db) / r


def peak(x, target_db):
    return x * db(target_db) / (np.max(np.abs(x)) + 1e-12)


# ------------------------------------------------------------ dialogue
def load_vo(path):
    return np.fromfile(path, dtype=np.float32) if path.endswith(".f32") else _wav(path)


def _wav(path):
    import wave
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768


def clip(vo, src_in, src_out, pre=0.04, post=0.06, fin=0.015, fout=0.04):
    return fade(vo[secs(src_in - pre): secs(src_out + post)], fin, fout)


# (character, subtitle text, source in, source out, timeline start, level dB, pan)
# Source times are positions in the raw recording; chosen takes noted.
DIALOGUE = [
    # --- Scene 1: family -------------------------------------------------
    ("JOHNNY", "Heyyyyy!", 6.01, 7.09, 2.55, -20, -0.15),                       # take 2 (longer stretch)
    ("GRANDMA", "Hiiiiiii sweetheart!", 12.60, 13.98, 2.85, -20, 0.15),         # take 2, overlaps Johnny
    ("JOHNNY", "Grandma / how are you?", 17.72, 19.00, 4.55, -20, 0),
    ("GRANDMA", "Good good…", 21.53, 22.83, 6.05, -20, 0),
    ("GRANDMA", "You got a / girlfriend yet?", 25.85, 28.11, 8.25, -19, 0),      # take 1: slow, deadly-serious read
    # --- Scene 2: someone you barely know ---------------------------------
    ("CRUSH", "Hey Johnny!", 40.00, 40.65, 14.35, -20, 0),
    ("JOHNNY", "Oh hey! What's u-", 44.93, 45.88, 15.55, -20, 0),
    ("JOHNNY", "oh wait / one sec-", 46.17, 47.88, 17.10, -21, 0),
    ("CRUSH", "Are you okay?", 53.10, 54.80, 20.25, -20, 0),
    ("JOHNNY", "YEAH I'M GOOD HAHAHA", 60.16, 62.20, 22.25, -17, 0),            # take 2 (no false start)
    # --- Scene 3: new friend ----------------------------------------------
    ("FRIEND", "Sooo / how'd it go?", 66.32, 67.71, 27.10, -20, 0),
    ("JOHNNY", "Yeah / I don't think / she's ever / calling me again.", 72.76, 74.67, 28.85, -20, 0),
    ("FRIEND", "Aww / don't worry buddy,", 78.83, 80.14, 31.35, -20, 0),
    ("FRIEND", "there's plenty / of fish / in the sea!", 81.23, 83.74, 32.80, -19, 0),
    ("JOHNNY", "Uhh… thanks", 87.97, 88.95, 36.10, -21, 0),                    # take 1
    # --- Scene 4: best friend ---------------------------------------------
    ("BERT", "What are you looking at?", 100.31, 100.91, 43.20, -21, 0),
    ("JOHNNY", "What are you / looking at?", 103.07, 103.79, 44.30, -21, 0),
    ("BERT", "I dunno / I thought you / were looking / at something", 112.91, 114.84, 45.75, -21, 0),
]

# Verbatim captions (every stutter/repeat as spoken, each stutter its own caption), one list per DIALOGUE
# entry: (caption text, source start, source end). Times are measured on the
# raw recording (word timestamps checked against voice-onset bursts), so
# captions stay locked to the words. Punctuation follows the script.
CAPTIONS = [
    [("Heyyyyy!", 6.01, 6.99)],
    [("Hiiiiiii", 12.63, 13.25), ("sweetheart!", 13.34, 13.98)],
    [("Gr-", 17.72, 17.95), ("Grandma", 17.99, 18.34), ("how are you?", 18.42, 19.00)],
    [("Good good…", 21.54, 22.83)],
    [("You got-", 25.86, 26.10), ("you got-", 26.33, 27.04),
     ("you got a", 27.25, 27.56), ("girlfriend yet?", 27.56, 28.11)],
    [("Hey Johnny!", 40.05, 40.65)],
    [("Oh hey!", 44.92, 45.44), ("What's u-", 45.52, 45.88)],
    [("Oh-", 46.16, 46.40), ("oh wait", 46.53, 47.01), ("one sec-", 47.12, 47.88)],
    [("Are you-", 53.11, 53.37), ("are you-", 53.60, 53.90), ("are you okay?", 54.15, 54.80)],
    [("YEAH", 60.17, 60.38), ("YEAH", 60.42, 60.77), ("I'M GOOD", 60.84, 61.67), ("HAHAHA", 61.73, 62.20)],
    [("Sooo", 66.38, 66.90), ("how'd it go?", 66.92, 67.71)],
    [("Yeah", 72.77, 73.05), ("I don't think", 73.10, 73.42), ("she's ever", 73.42, 73.78),
     ("calling me again", 73.78, 74.67)],
    [("Aww", 78.82, 79.05), ("don't worry buddy", 79.12, 80.14)],
    [("There's plenty", 81.30, 82.02), ("of fish", 82.02, 82.68), ("in the SEA!", 82.68, 83.74)],
    [("Uhh…", 87.96, 88.29), ("thanks", 88.47, 88.95)],
    [("What are you", 100.33, 100.60), ("looking at?", 100.60, 100.91)],
    [("What are you", 103.12, 103.41), ("looking at?", 103.41, 103.79)],
    [("I dunno", 112.90, 113.30), ("I dunno", 113.35, 113.76), ("I thought you", 113.82, 114.10),
     ("were looking", 114.10, 114.34), ("at something", 114.34, 114.84)],
]
assert len(CAPTIONS) == 18

DUR = 49.5

# Scene boundaries (timeline seconds) for the animator / cue sheet
SCENES = [
    ("FaceTime loading -> Calling with family", 0.0, 11.55),
    ("Calling with someone you barely know", 11.55, 24.85),
    ("Calling with a new friend", 24.85, 38.1),
    ("Calling with best friend", 38.1, DUR),
]


def main(vo_path, out_dir):
    vo = load_vo(vo_path)
    vo = hp_fast(vo, 85)  # rumble cleanup
    dlg, sfx, mus = Track(DUR), Track(DUR), Track(DUR)
    cues = []

    for (who, text, a, b, at, lvl, pan), caps in zip(DIALOGUE, CAPTIONS):
        c = clip(vo, a, b)
        # Hard cut-offs for interrupted lines (no tail)
        if text.endswith("-"):  # interrupted line
            c = fade(vo[secs(a - 0.04): secs(b + 0.01)], 0.015, 0.012)
        # Levelled on the voiced part only
        body = vo[secs(a): secs(b)]
        g = db(lvl) / (np.sqrt(np.mean(body ** 2)) + 1e-9)
        dlg.add(c * g, at - 0.04, pan=pan)
        # " / " in text = forced caption break (see build_subs.py)
        cues.append({"who": who, "text": text, "start": round(at, 3),
                     "end": round(at + (b - a), 3),
                     "chunks": [[t, round(at + max(cs, a) - a, 3), round(at + min(ce, b) - a, 3)]
                                for t, cs, ce in caps]})

    # All effects and music are real recordings from free libraries (see
    # assets/CREDITS.md). Levels: SFX peaks set in dBFS, beds RMS-levelled.
    FT = "yt_facetime.wav"      # YouTube: "Facetime Connecting and Disconnecting Sound Effect"

    def ring(at, dur):          # FaceTime outgoing-call ripple, repeated while it rings
        t = 0.0
        while t < dur - 0.3:
            sfx.put(peak(smp(FT, 0.30, min(0.85, dur - t), 0.005, 0.06), -15), at + t)
            t += 1.25

    def connect(at):            # FaceTime connect
        sfx.put(peak(smp(FT, 2.00, 0.75, 0.005, 0.08), -14), at)

    def hang_up(at):            # FaceTime disconnect
        sfx.put(peak(smp(FT, 6.52, 0.60, 0.005, 0.08), -15), at)

    def swoosh(at, k=1):        # screen change
        sfx.put(peak(smp(f"oc_SWSH_Swoosh_OS_0{k}.ogg"), -18), at)

    def room(at, dur, g=-50):
        sfx.put(rms_to(smp("oc_AMBHome_RoomTone_LOOP_01.ogg", 5, dur, 0.3, 0.3), g), at)

    # ---- Scene 1: FaceTime loading -> family --------------------------------
    ring(0.15, 2.0)
    connect(2.1)
    room(2.2, 11.55 - 2.2)
    mus.put(rms_to(smp("fs_mus_ukulele.mp3", 0.0, 5.25, 0.2, 0.02), -27), 2.3)   # happy bed...
    # ...cut dead by a cinematic boom the moment Grandma turns serious
    sfx.put(peak(smp("yt_cinematic_boom.wav", 0.10, 1.25, 0.003, 0.5), -8), 7.4)
    hang_up(11.0)

    # ---- Scene 2: crush -----------------------------------------------------
    swoosh(11.3, 1)
    ring(11.7, 2.2)
    sfx.put(rms_to(smp("fs_hair_ruffle.mp3", 4.4, 2.4, 0.1, 0.3), -34), 11.65)     # fixing hair
    connect(13.85)
    room(14.0, 24.85 - 14.0)
    mus.put(rms_to(smp("fs_mus_waltz.mp3", 0.0, 3.2, 0.1, 0.03), -29), 14.0)    # cut on the knock
    knock = 15.55 + (45.88 - 44.93) + 0.02
    sfx.put(peak(smp("fs_phone_drop_table.mp3"), -4), knock)                 # phone hits the table
    sfx.put(peak(smp("fsd_dice-throw.wav"), -14), knock + 0.04)              # ...and skids
    # pure chaos: it bangs around while he grabs at it
    sfx.put(peak(smp("fs_phone_drops_multi.mp3", 14.9, 2.4, 0.01, 0.2), -9), knock + 0.55)
    sfx.put(peak(smp("fs_phone_handling.mp3", 3.3, 2.1, 0.02, 0.2), -13), knock + 0.4)
    sfx.put(peak(smp("fs_phone_on_wood.mp3"), -9), knock + 2.55)
    sfx.put(peak(smp("fs_phone_handling.mp3", 11.2, 0.8, 0.02, 0.2), -12), knock + 2.75)
    sfx.put(peak(smp("fsd_wood-knock-hit.wav"), -14), 21.98)                 # propped back up
    hang_up(24.6)

    # ---- Scene 3: new friend (walking down the street) ---------------------
    swoosh(24.8, 2)
    ring(25.0, 1.6)
    connect(26.5)
    sfx.put(rms_to(smp("fs_street.mp3", 2.0, 38.1 - 26.6, 0.4, 0.4), -38), 26.6)
    steps = np.concatenate([smp("fs_footsteps.mp3", 10.0, 7.0, 0.05, 0.05),
                            smp("fs_footsteps.mp3", 28.8, 4.2, 0.05, 0.3)])
    sfx.put(rms_to(steps, -40), 26.8)
    mus.put(rms_to(smp("fs_mus_lofi.mp3", 0.0, 35.45 - 26.7, 0.5, 0.25), -31), 26.7)
    # music drops out for the cringe beat before "Uhh... thanks"
    hang_up(37.55)

    # ---- Scene 4: best friend (silence for ages) ---------------------------
    swoosh(37.95, 3)
    ring(38.3, 1.6)
    connect(39.8)
    room(40.0, DUR - 40.0, -46)
    tv = smp("oc_CRWDConv_Restaurant_LOOP_01.ogg", 10, DUR - 40.0, 0.4, 0.4)
    tv = np.stack([bp_fast(tv[:, 0], 250, 2500), bp_fast(tv[:, 1], 250, 2500)], 1)
    sfx.put(rms_to(tv, -46), 40.0)                                    # muffled TV across the room
    sfx.put(rms_to(smp("fs_chew_peanuts.mp3", 1.5, 3.4, 0.05, 0.2), -31), 40.0)   # Bert chewing
    sfx.put(rms_to(smp("fs_chew_peanuts2.mp3", 2.8, 1.2, 0.05, 0.2), -35), 41.6)  # Johnny chewing
    sfx.put(rms_to(smp("fs_chew_peanuts.mp3", 4.2, 1.5, 0.05, 0.3), -33), 47.9)   # back to chewing

    # ---- duck music under dialogue (sidechain-style) -------------------------
    dmono = np.abs(dlg.L) + np.abs(dlg.R)
    envd = lp_fast(dmono, 6)
    duck = 1 - 0.45 * np.clip(envd / (envd.max() * 0.15), 0, 1)
    mus.L *= duck
    mus.R *= duck

    L = dlg.L + sfx.L + mus.L
    R = dlg.R + sfx.R + mus.R
    master = np.stack([L, R], 1)
    master *= db(-1.0) / np.max(np.abs(master))  # peak-normalise; LUFS done in ffmpeg

    import wave
    for name, data in [("mix_premaster", master),
                       ("stem_dialogue", np.stack([dlg.L, dlg.R], 1) * db(-1.0) / np.max(np.abs(master))),
                       ("stem_sfx", np.stack([sfx.L, sfx.R], 1) * db(-1.0) / np.max(np.abs(master))),
                       ("stem_music", np.stack([mus.L, mus.R], 1) * db(-1.0) / np.max(np.abs(master)))]:
        with wave.open(f"{out_dir}/{name}.wav", "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())

    json.dump({"duration": DUR, "scenes": SCENES, "cues": cues},
              open(f"{out_dir}/cues.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

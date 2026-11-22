"""Make the .ass subtitle file from cues.json (+ optional word timings).

Usage: python3 build_subs.py <cues.json> <out.ass> [word_times.json]

Style knobs are all in STYLE below so they can be matched to the reference.
"""
import json
import re
import sys

W, H = 2160, 3840
# Matched to the Blobby reference Short (vidIQ frame analysis): bold geometric
# sans, white with a solid black outline, no box/shadow, centred near the TOP
# of frame (~14% down), 1-3 words per caption, one line at a time, hard cuts
# with no animation, captions clear during pauses, commas/full stops dropped
# ("?" kept, no "!"), exclaimed/shouted words in ALL CAPS (set in the cue text).
STYLE = dict(
    font="Montserrat",
    bold=-1,
    size=180,             # px at 2160x3840 -> cap height ~3.3% of frame height
    primary="&H00FFFFFF",  # white
    outline_col="&H00000000",
    outline=8,            # ~5-6% of letter height
    shadow=0,
    align=5,              # middle-centre anchor, placed with \pos
    y=540,                # ~14% from the top
    max_words=3,
    max_chars=16,
    min_dur=0.22,
    join_gap=0.25,        # gaps shorter than this are bridged; longer ones go blank
)


# Johnny (main character) keeps the default white in every skit; everyone
# else gets an easy-to-read colour that suits them (RGB hex).
SPEAKER_COLOURS = {
    "JOHNNY": "FFFFFF",   # white (default)
    "GRANDMA": "FFD166",  # warm golden yellow - cosy grandma
    "CRUSH": "FF9EC7",    # soft pink - the crush
    "FRIEND": "7DF0A8",   # mint green - upbeat, corny friend
    "BERT": "8CCBFF",     # sky blue - laid-back best friend
}


def ass_colour(rgb):
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def ts(t):
    t = max(0.0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def clean(w):
    """Drop commas, full stops and '!' (exclaimed words are written in CAPS
    instead); keep '?', '…' and cut-off '-'."""
    w = w.replace("!", "")
    if w.endswith("-") or w.endswith("…"):
        return w
    return re.sub(r"[,.]+$", "", w)


def chunk_words(words, max_words, max_chars):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[?!.…,]$", w):  # natural phrase end
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def main(cues_path, out_path, words_path=None):
    data = json.load(open(cues_path))
    wt = json.load(open(words_path)) if words_path else {}
    st = STYLE
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {W}",
        f"PlayResY: {H}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Sub,{st['font']},{st['size']},{st['primary']},&H000000FF,{st['outline_col']},&H00000000,"
        f"{st['bold']},0,0,0,100,100,0,0,1,{st['outline']},{st['shadow']},{st['align']},120,120,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events = []  # [start, end, text, who]
    for i, c in enumerate(data["cues"]):
        if c.get("chunks"):  # verbatim captions with measured timings
            events += [[s, e, " ".join(clean(w) for w in t.split()), c["who"]]
                       for t, s, e in c["chunks"]]
            continue
        # " / " in the cue text marks a preferred phrase break
        words = [w for w in c["text"].split() if w != "/"]
        chunks = [ch for part in c["text"].split(" / ")
                  for ch in chunk_words(part.split(), st["max_words"], st["max_chars"])]
        times = wt.get(str(i))  # [start, end] per script word, timeline secs
        dur = c["end"] - c["start"]
        total = sum(len(w) for w in words)
        n = acc = 0
        for k, ch in enumerate(chunks):
            if times and len(times) == len(words):
                s, e = times[n][0], times[n + len(ch) - 1][1]
            else:
                s = c["start"] + dur * acc / total
                acc += sum(len(w) for w in ch)
                e = c["start"] + dur * acc / total
            if k == 0:
                s = c["start"]
            if k == len(chunks) - 1:
                e = c["end"]
            n += len(ch)
            events.append([s, e, " ".join(clean(w) for w in ch), c["who"]])

    # one caption on screen at a time: hard cut to the next caption,
    # bridge tiny gaps, leave real pauses blank
    events.sort(key=lambda e: e[0])
    for a, b in zip(events, events[1:]):
        a[1] = max(a[1], a[0] + st["min_dur"])
        if b[0] - a[1] < st["join_gap"]:
            a[1] = b[0]
        a[1] = min(a[1], b[0])
    events[-1][1] = max(events[-1][1], events[-1][0] + st["min_dur"])

    for s, e, text, who in events:
        col = ass_colour(SPEAKER_COLOURS.get(who, "FFFFFF"))
        lines.append(f"Dialogue: 0,{ts(s)},{ts(e)},Sub,{who},0,0,0,,"
                     f"{{\\pos({W // 2},{st['y']})\\1c{col}&}}{text}")
    open(out_path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

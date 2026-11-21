"""Make the .ass subtitle file from cues.json (+ optional word timings).

Usage: python3 build_subs.py <cues.json> <out.ass> [word_times.json]

Style knobs are all in STYLE below so they can be matched to the reference.
"""
import json
import re
import sys

W, H = 2160, 3840
STYLE = dict(
    font="Montserrat ExtraBold",
    size=180,            # px at 2160x3840
    primary="&H00FFFFFF",  # white text
    outline_col="&H00000000",
    outline=13,
    shadow=0,
    align=5,             # 5 = middle-centre (position set with \pos)
    y=2560,              # ~2/3 down the frame
    max_chars=18,        # per caption chunk
)


def ts(t):
    t = max(0.0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def chunk_words(words, max_chars):
    chunks, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > max_chars:
            chunks.append(cur)
            cur = []
        cur.append(w)
        # break after sentence punctuation
        if re.search(r"[?!.…,]$", w) and len(" ".join(cur)) > 6:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    # avoid orphan single short words at the end
    if len(chunks) > 1 and len(" ".join(chunks[-1])) <= 4:
        last = chunks.pop()
        chunks[-1] += last
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
        f"0,0,0,0,100,100,0,0,1,{st['outline']},{st['shadow']},{st['align']},120,120,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events = []  # (start, end, text, row)
    for i, c in enumerate(data["cues"]):
        # " / " in the cue text forces a caption break; otherwise auto-chunk
        words = [w for w in c["text"].split() if w != "/"]
        overlapped = any(o is not c and o["who"] != c["who"] and o["start"] < c["end"]
                         and c["start"] < o["end"] for o in data["cues"])
        if overlapped:
            chunks = [words]  # simultaneous speakers: keep each line whole, stacked
        elif " / " in c["text"]:
            chunks = [p.split() for p in c["text"].split(" / ")]
        else:
            chunks = chunk_words(words, st["max_chars"])
        times = wt.get(str(i))  # list of [start, end] per script word, timeline secs
        n = 0
        dur = c["end"] - c["start"]
        total = sum(len(w) for w in words)
        acc = 0
        for k, ch in enumerate(chunks):
            if times and len(times) == len(words):
                s = times[n][0]
                e = times[n + len(ch) - 1][1]
            else:
                s = c["start"] + dur * acc / total
                acc += sum(len(w) for w in ch)
                e = c["start"] + dur * acc / total
            if k == 0:
                s = c["start"]
            if k == len(chunks) - 1:
                e = c["end"]
            n += len(ch)
            events.append([s, e, " ".join(ch), c["who"]])

    # chain consecutive chunks so captions don't flicker; hold short ends
    events.sort(key=lambda e: e[0])
    for e in events:
        e[1] = max(e[1], e[0] + 0.35)
    for a, b in zip(events, events[1:]):
        if a[3] == b[3] and b[0] - a[1] < 0.25:
            a[1] = b[0]

    # overlapping speakers (scene 1) -> stack: earlier speaker on top
    for i, e in enumerate(events):
        row = 0
        for p in events[:i]:
            if p[0] < e[1] and e[0] < p[1] and p[3] != e[3]:
                row = 1
        y = st["y"] + row * int(st["size"] * 1.25)
        if row:
            # push the earlier caption up instead of the later one down
            y = st["y"] + int(st["size"] * 0.63)
            for p in events[:i]:
                if p[0] < e[1] and e[0] < p[1]:
                    p.append(st["y"] - int(st["size"] * 0.63))
        e.append(y)

    for s, e, text, who, *ys in events:
        y = ys[-1]
        lines.append(f"Dialogue: 0,{ts(s)},{ts(e)},Sub,{who},0,0,0,,{{\\pos({W // 2},{y})}}{text}")
    open(out_path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

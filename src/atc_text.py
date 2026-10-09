"""ATC transcript normalisation and a rule-based clearance parser.

Normalisation maps Whisper-style output ("Lufthansa 4AB, runway 27") onto the ATCO2/UWB
transcription convention (lower-case words, digits and letters spoken one by one), so WER
measures recognition errors rather than formatting differences.

The parser turns one transmission into structured intent: callsign, runway, and the
safety-relevant instructions (hold short, cross, line up, cleared takeoff/land, taxi via)."""
import re

DIGITS = "zero one two three four five six seven eight nine".split()
SPOKEN_DIGIT = {w: str(i) for i, w in enumerate(DIGITS)} | {"niner": "9", "tree": "3", "fife": "5", "fower": "4"}
NATO = ("alfa bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa "
        "quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
LETTER = {w[0]: w for w in NATO}
SPOKEN_LETTER = {w: w[0] for w in NATO} | {"alpha": "a", "juliet": "j", "x-ray": "x", "whisky": "w"}
TENS = {"thousand": "000", "hundred": "00"}

# telephony designators seen in the corpora (lower-case spoken form)
AIRLINES = [a.strip() for a in (
    "lufthansa|speedbird|easy|ryanair|wizzair|wizz air|czech|csa|skytravel|sky travel|austrian|swiss|klm|"
    "air france|delta|united|american|emirates|qatar|turkish|aeroflot|lot|polish|smartwings|travel service|"
    "germanwings|german wings|eurowings|condor|brussels|finnair|nordic|scandinavian|iberia|alitalia|jet two|"
    "topswiss|sky shuttle|el al|dynasty|shamrock|tap|aurela|belavia|transavia|dolomiti|tarom|cactus|jetblue|"
    "southwest|spirit|frontier|alaska|envoy|piedmont|republic|endeavor|skywest|air canada|jazz|atlas|fedex|ups"
).split("|")]
KEEP_CAPS = {"csa", "klm", "lot", "tap", "ups", "ok"}


def normalise(text: str) -> str:
    words = []
    for raw in re.split(r"[\s,;:!?/()\"]+", text.replace("-", " ")):
        w = raw.strip(".'")
        if not w:
            continue
        mixed = any(c.isdigit() for c in w) and any(c.isalpha() for c in w)
        caps = w.isalpha() and w.isupper() and 1 < len(w) <= 6 and w.lower() not in KEEP_CAPS
        if mixed or caps:  # "4AB", "K6", "1C", "FAO" -> spoken character by character
            for c in w.lower():
                if c.isdigit(): words.append(DIGITS[int(c)])
                elif c in LETTER: words.append(LETTER[c])
            continue
        if re.fullmatch(r"\d+(\.\d+)?", w):
            words += [DIGITS[int(c)] if c.isdigit() else "decimal" for c in w]
            continue
        w = re.sub(r"[^a-z']", "", w.lower())
        w = {"alpha": "alfa", "juliet": "juliett", "x": "xray", "whisky": "whiskey"}.get(w, w)
        if w:
            words.append(w)
    return " ".join(words)


def _tokens(text):
    return normalise(text).split()


RUNWAY_SIDE = {"left": "L", "right": "R", "center": "C", "centre": "C"}


def _read_number(toks, i, max_len=4):
    """Read spoken digits starting at toks[i]; returns (string, next index)."""
    s = ""
    while i < len(toks) and toks[i] in SPOKEN_DIGIT and len(s) < max_len:
        s += SPOKEN_DIGIT[toks[i]]; i += 1
    return s, i


def parse(text: str) -> dict:
    toks = _tokens(text)
    out = {"callsign": None, "runway": None, "hold_short": False, "cross": False, "line_up": False,
           "cleared_takeoff": False, "cleared_land": False, "taxi": False, "via": []}
    joined = " ".join(toks)
    # callsign: airline telephony + digits/letters, or a spoken registration (5 letters)
    for name in sorted(AIRLINES, key=len, reverse=True):
        m = re.search(rf"\b{name}\b", joined)
        if m and m.start() > 40:  # telephony is said first; a later match is usually something else
            m = None
        if m:
            rest = joined[m.end():].split()
            tail, j = "", 0
            while j < len(rest) and (rest[j] in SPOKEN_DIGIT or rest[j] in SPOKEN_LETTER) and len(tail) < 6:
                tail += SPOKEN_DIGIT.get(rest[j]) or SPOKEN_LETTER[rest[j]].upper(); j += 1
            if tail and tail[0].isdigit():
                out["callsign"] = name.replace(" ", "").upper() + tail
                break
    if out["callsign"] is None:  # spoken registration, e.g. "oscar kilo foxtrot alfa oscar"
        run = []
        for w in toks + ["<end>"]:
            if (w in SPOKEN_LETTER or w in SPOKEN_DIGIT) and len(run) < 6:
                run.append(SPOKEN_LETTER.get(w, "").upper() or SPOKEN_DIGIT[w])
                continue
            if len(run) >= 4 and sum(c.isalpha() for c in run) >= 3:
                out["callsign"] = "".join(run); break
            run = [SPOKEN_LETTER.get(w, "").upper() or SPOKEN_DIGIT[w]] if (w in SPOKEN_LETTER or w in SPOKEN_DIGIT) else []
    out["taxi"] = bool(re.search(r"\btaxi\b", joined))
    # runway
    for i, w in enumerate(toks):
        if w == "runway":
            num, j = _read_number(toks, i + 1, 2)
            if num:
                nxt = toks[j] if j < len(toks) else ""
                side = RUNWAY_SIDE.get(nxt) or {"lima": "L", "romeo": "R", "charlie": "C"}.get(nxt, "")
                out["runway"] = num.zfill(2) + side
                break
    out["hold_short"] = bool(re.search(r"\bhold(ing)? short\b", joined)) or (
        bool(re.search(r"\bholding (point|position)\b", joined)) and out["taxi"])
    out["cross"] = bool(re.search(r"\bcross(ing)? runway\b", joined))
    out["line_up"] = bool(re.search(r"\bline up\b|\blining up\b", joined))
    out["cleared_takeoff"] = bool(re.search(r"\bcleared (for )?take ?off\b", joined))
    out["cleared_land"] = bool(re.search(r"\bcleared (to )?land\b", joined))
    m = re.search(r"\bvia ((?:[a-z]+ ?){1,8})", joined)
    if m:
        via = []
        for w in m.group(1).split():
            if w in SPOKEN_LETTER: via.append(SPOKEN_LETTER[w].upper())
            elif w in SPOKEN_DIGIT and via: via[-1] += SPOKEN_DIGIT[w]
            else: break
        out["via"] = via
    return out


if __name__ == "__main__":
    for s in ["oscar kilo foxtrot alfa oscar taxi to holding point runway two seven via alfa charlie",
              "Lufthansa 4AB, hold short runway 1C, cross runway 30 via K6",
              "speedbird two two seven line up and wait runway one nine left",
              "csa one two three cleared for takeoff runway two four"]:
        print(normalise(s)); print("  ", parse(s))

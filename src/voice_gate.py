"""What ASR quality means operationally: false alerts from authorised runway entries.

A runway-conformance monitor alerts whenever a target enters a runway's protected area.
Most entries are authorised (line up and wait, cleared for takeoff, cross runway X). With
voice, the monitor suppresses an entry when the parsed clearance names the right callsign,
runway and instruction. Joint parse accuracy is measured per transmission on real ATC
audio (ASR output vs human transcript, both through the same parser); the number of
authorised entries per hour is counted from the Oct 1 KIAD test day."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from atc_text import parse
from tracks import TEST_START

ROOT = Path(__file__).resolve().parents[1]
CMDS = ["hold_short", "cross", "line_up", "cleared_takeoff", "cleared_land"]


def runway_entries(d):
    """Ground entries into a runway strip from the taxiway side: the target was on the ground
    and off the runway, then stays on the runway >= 5 s. Landings (first ground fix already on
    the runway) and edge jitter are not entries."""
    ents = []
    g_all = d[(d.t >= TEST_START) & d.sensor.isin(["adsb", "tisb"])].sort_values(["track_id", "t"])
    for tid, g in g_all.groupby("track_id"):
        on = (g.rwy_dist.values < 30) & g.ground.values
        off_gnd = (g.rwy_dist.values >= 30) & g.ground.values
        t = g.t.values
        was_off = False
        i = 0
        while i < len(g):
            if off_gnd[i]:
                was_off = True
            elif on[i] and was_off:
                j = i
                while j + 1 < len(g) and on[j + 1]: j += 1
                if t[j] - t[i] >= 5:
                    ents.append(g.iloc[i]); was_off = False
                i = j
            elif not g.ground.values[i]:
                was_off = False
            i += 1
    ent = pd.DataFrame(ents)
    return ent, (d.t.max() - TEST_START) / 3600


def joint_accuracy(csv):
    df = pd.read_csv(csv).fillna("")
    rows = []
    for r, h in zip(df.ref, df.hyp):
        pr, ph = parse(r), parse(h)
        if not any(pr[k] for k in CMDS + ["runway"]):
            continue  # only transmissions that carry a runway or a safety-relevant instruction
        ok = pr["callsign"] == ph["callsign"] and pr["runway"] == ph["runway"] and all(pr[k] == ph[k] for k in CMDS)
        rows.append(ok)
    return float(np.mean(rows)), len(rows)


def main():
    d = pd.read_parquet(ROOT / "data/tracks.parquet")
    ent, hours = runway_entries(d)
    per_hour = len(ent) / hours
    out = {"authorised_entries": int(len(ent)), "hours": hours, "entries_per_hour": per_hour,
           "no_voice_false_alerts_per_hour": per_hour}
    for tag in sorted({p.name.split("_")[1] for p in (ROOT / "reports").glob("asr_*_atco2.csv")}):
        for split in ["uwb_test", "atco2"]:
            p = ROOT / f"reports/asr_{tag}_{split}.csv"
            if not p.exists(): continue
            acc, n = joint_accuracy(p)
            out[f"{tag}/{split}"] = {"joint_parse_accuracy": acc, "n_transmissions": n,
                                     "false_alerts_per_hour": per_hour * (1 - acc)}
            print(f"{tag:28s} {split:9s} joint parse acc {acc:.3f} (n={n}) -> {per_hour * (1 - acc):.1f} false alerts/h", flush=True)
    print(f"{len(ent)} authorised runway entries in {hours:.1f} h = {per_hour:.1f}/h (all false alerts without voice)")
    json.dump(out, open(ROOT / "reports/voice_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()

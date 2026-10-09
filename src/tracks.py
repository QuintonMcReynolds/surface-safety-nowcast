"""Raw ADS-B Exchange reports -> de-duplicated per-target tracks with hindsight activity labels.

readsb-hist snapshots repeat a target's last position until a new one arrives, so the true
fix time is now - seen_pos and repeated fixes are dropped. Labels are assigned offline with
the whole track visible (future included); online models never see the future, which is
what makes the labels a fair grading key."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from airport import Airport, to_xy

ROOT = Path(__file__).resolve().parents[1]
STATES = ["STOPPED", "TAXI", "HOLD_SHORT", "RWY_SLOW", "TAKEOFF_ROLL", "LANDING_ROLL", "AIRBORNE"]
S = {s: i for i, s in enumerate(STATES)}
FT = 0.3048
# Train on the Aug 1 and Sep 1 2025 sample days, test on Oct 1 (12:00-18:00 UTC each day).
TEST_START = 1759320000.0  # 2025-10-01 12:00:00 UTC


def build(raw_paths=None, ap=None):
    ap = ap or Airport()
    raw_paths = raw_paths or sorted((ROOT / "data/raw").glob("KIAD_*_12-18Z.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in raw_paths], ignore_index=True)
    d["t"] = d["now"] - d["seen_pos"].fillna(0)
    d = d.sort_values(["hex", "t"])
    d = d.drop_duplicates(["hex", "t"])
    d["ground"] = d["alt_baro"].eq("ground")
    d["alt_ft"] = pd.to_numeric(d["alt_baro"], errors="coerce").fillna(0.0).where(~d["ground"], 0.0)
    d["x"], d["y"] = to_xy(d.lat.values, d.lon.values)
    d["gs"] = d["gs"].astype(float)
    d["sensor"] = np.where(d["type"].str.startswith("tisb"), "tisb",
                  np.where(d["type"].isin(["adsb_icao", "adsb_other", "adsr_icao"]), "adsb", "other"))
    d["vehicle"] = d["category"].isin(["C1", "C2", "C3"])
    # split a target into tracks at gaps > 120 s
    d["gap"] = d.groupby("hex")["t"].diff().fillna(1e9) > 120
    d["track_id"] = d["hex"] + "_" + d.groupby("hex")["gap"].cumsum().astype(str)
    # runway / hold context
    name, rdist, along = ap.runway_context(d.x.values, d.y.values)
    d["rwy"], d["rwy_dist"], d["rwy_along"] = name, rdist, along
    hi, hd = ap.nearest_hold(d.x.values, d.y.values)
    d["hold_idx"], d["hold_dist"] = hi, hd
    hdg = np.array([ap.runways[n].heading for n in name])
    trk = d["track"].astype(float).fillna(0).values
    diff = np.abs((trk - hdg + 180) % 360 - 180)
    d["rwy_align"] = np.minimum(diff, 180 - diff)  # 0 = parallel to the runway either way
    d = pd.concat([label_track(g) for _, g in d.groupby("track_id", sort=False)])
    return d.reset_index(drop=True)


def label_track(g):
    g = g.copy()
    n = len(g)
    t = g.t.values
    gs = pd.Series(g.gs.values).rolling(3, center=True, min_periods=1).median().values
    gs = np.nan_to_num(gs, nan=0.0)
    air = (~g.ground.values) & (g.alt_ft.values > 0)
    on_rwy = (g.rwy_dist.values < 30) & ~air
    lab = np.full(n, S["TAXI"])
    lab[gs < 1.5] = S["STOPPED"]
    hold = (gs < 1.5) & (g.hold_dist.values < 90) & ~on_rwy
    lab[hold] = S["HOLD_SHORT"]
    lab[on_rwy] = S["RWY_SLOW"]
    lab[air] = S["AIRBORNE"]
    fast = on_rwy & (gs >= 35)
    # takeoff roll: fast on runway and airborne within 90 s afterwards; landing: airborne within 90 s before
    air_t = t[air]
    for i in np.where(fast)[0]:
        after = air_t[(air_t > t[i]) & (air_t < t[i] + 90)]
        before = air_t[(air_t < t[i]) & (air_t > t[i] - 90)]
        if len(after) and (not len(before) or after.min() - t[i] < t[i] - before.max()):
            lab[i] = S["TAKEOFF_ROLL"]
        elif len(before):
            lab[i] = S["LANDING_ROLL"]
        else:
            lab[i] = S["RWY_SLOW"] if gs[i] < 60 else S["TAKEOFF_ROLL"] if np.all(np.diff(gs[max(0, i-3):i+1]) >= -2) else S["LANDING_ROLL"]
    # takeoff roll starts where the aircraft begins accelerating on the runway: extend backwards
    for i in np.where(lab == S["TAKEOFF_ROLL"])[0]:
        j = i - 1
        while j >= 0 and lab[j] == S["RWY_SLOW"] and gs[j] >= 5 and gs[j] <= gs[j + 1] + 1:
            lab[j] = S["TAKEOFF_ROLL"]; j -= 1
    g["state"] = lab
    g["gs_s"] = gs
    return g


def flight_events(d):
    """One row per track: departure / arrival / crossing events with times (hindsight)."""
    rows = []
    for tid, g in d.groupby("track_id"):
        st, t = g.state.values, g.t.values
        r = {"track_id": tid, "hex": g.hex.iloc[0], "flight": g.flight.dropna().iloc[0] if g.flight.notna().any() else None,
             "sensor": g.sensor.mode().iloc[0], "vehicle": bool(g.vehicle.any()), "n": len(g),
             "t0": t[0], "t1": t[-1]}
        to = np.where(st == S["TAKEOFF_ROLL"])[0]
        if len(to) and (st[:to[0]] != S["AIRBORNE"]).all() and to[0] > 3:
            r["roll_start"] = t[to[0]]; r["dep_rwy"] = g.rwy.values[to[0]]
            r["dep_rwy_enter"] = t[np.where((g.rwy_dist.values < 30) & (np.arange(len(g)) <= to[0]))[0][-1::-1]]
            # first point of the final continuous on-runway stretch before the roll
            k = to[0]
            while k > 0 and g.rwy_dist.values[k - 1] < 30 and st[k - 1] != S["AIRBORNE"]:
                k -= 1
            r["dep_rwy_enter"] = t[k]
        la = np.where(st == S["LANDING_ROLL"])[0]
        if len(la):
            r["touchdown"] = t[la[0]]; r["arr_rwy"] = g.rwy.values[la[0]]
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build()
    d.to_parquet(ROOT / "data/tracks.parquet")
    ev = flight_events(d)
    ev.to_parquet(ROOT / "data/events.parquet")
    d["day"] = pd.to_datetime(d.t, unit="s").dt.strftime("%Y-%m-%d")
    print(d.groupby("day").track_id.nunique().to_dict(), "tracks per day")
    print(len(d), "fixes,", d.track_id.nunique(), "tracks")
    print(d.groupby("sensor").track_id.nunique().to_dict())
    print(pd.Series(d.state.map(dict(enumerate(STATES)))).value_counts().to_dict())
    print("departures with roll start:", ev.roll_start.notna().sum(), "arrivals with touchdown:", ev.touchdown.notna().sum())
    print(ev.dep_rwy.value_counts().to_dict(), ev.arr_rwy.value_counts().to_dict())

"""Hold-short violation prediction: alert lead time vs false alerts, under degraded surveillance.

Events: every real KIAD approach to a runway hold line that ended in a stop (an instructed
hold, done correctly). Counterfactual twin: the same real track up to the moment braking
began, then continuing at its approach speed and heading, i.e. the aircraft fails to stop.
Truth for a twin is the time it enters the runway's protected area (|offset from centreline|
less than the hold line's offset). Real stops are the false-alert population.

Alerters: (1) geofence rule (near the line and still moving), (2) kinematic rule (required
deceleration to stop before the line exceeds what taxiing aircraft actually use), (3) gradient
boosted P(enter protected area within 20 s), thresholds chosen on training events."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent))
from airport import Airport
from activity import degrade
from tracks import S, TEST_START

ROOT = Path(__file__).resolve().parents[1]
KT = 0.514444
HORIZON = 20.0


def hold_offsets(ap):
    off = {}
    for h in ap.holds:
        al, cr = ap.runways[h.runway].along_cross(*h.xy)
        off[h.id] = abs(float(cr))
    return off


def find_events(d, ap):
    off = hold_offsets(ap)
    ev = []
    for tid, g in d[d.sensor.isin(["adsb", "tisb"])].groupby("track_id"):
        st = g.state.values
        starts = np.where((st == S["HOLD_SHORT"]) & (np.r_[-1, st[:-1]] != S["HOLD_SHORT"]))[0]
        for k in starts:
            if k < 4:
                continue
            ends = np.where(st[k:] != S["HOLD_SHORT"])[0]
            dur = (g.t.values[k + ends[0]] if len(ends) else g.t.values[-1]) - g.t.values[k]
            pre = g.iloc[max(0, k - 40):k + 1]
            pre = pre[pre.t >= g.t.values[k] - 120]
            if dur < 5 or pre.gs_s.max() < 4:
                continue
            hi = int(g.hold_idx.values[k]); h = ap.holds[hi]
            rw = ap.runways[h.runway]
            ev.append({"track_id": tid, "k": int(k), "sensor": g.sensor.iloc[0], "vehicle": bool(g.vehicle.any() or g.sensor.iloc[0] == "tisb"), "t_stop": float(g.t.values[k]), "hold": hi, "runway": h.runway,
                       "offset": off[h.id], "hold_s": float(dur)})
    return pd.DataFrame(ev)


def approach(g, e, ap):
    """Real approach segment (up to the stop + 20 s) and its no-stop twin."""
    t_stop = e.t_stop
    seg = g[(g.t >= t_stop - 120) & (g.t <= t_stop + 20)].copy()
    pre = seg[seg.t <= t_stop]
    v = pre.gs_s.values
    vmax = pd.Series(v).rolling(3, min_periods=1).max().values
    # braking starts at the last fix still near cruise-taxi speed
    k = int(np.where(v >= 0.8 * v.max())[0][-1])
    k = min(k, len(pre) - 2)
    p0 = pre[["x", "y"]].values[k]; p1 = pre[["x", "y"]].values[max(k - 2, 0)]
    head = p0 - p1
    if np.linalg.norm(head) < 1:
        trk = np.radians(pre.track.astype(float).fillna(0).values[k]); head = np.array([np.sin(trk), np.cos(trk)])
    head = head / np.linalg.norm(head)
    speed = max(v[k], 6.0) * KT
    t0 = pre.t.values[k]
    # keep the real sampling pattern: reuse the real time stamps after t0, extend at 5 s
    ts = list(seg.t.values[seg.t.values > t0])
    while not ts or ts[-1] < t0 + 90: ts.append((ts[-1] if ts else t0) + 5.0)
    ts = np.array(ts)
    xy = p0[None] + (ts - t0)[:, None] * speed * head[None]
    twin = pd.concat([pre.iloc[:k + 1], pd.DataFrame({"t": ts, "x": xy[:, 0], "y": xy[:, 1], "gs": v[k],
                                                      "track": np.degrees(np.arctan2(head[0], head[1])) % 360,
                                                      "ground": True})], ignore_index=True)
    twin["gs_s"] = twin.gs
    rw = ap.runways[e.runway]
    _, cr = rw.along_cross(twin.x.values, twin.y.values)
    inside = np.abs(cr) < e.offset
    if not inside.any() or inside[: k + 1].any():
        return seg, None, None
    t_cross = float(twin.t.values[np.argmax(inside)])
    twin = twin[twin.t <= t_cross + 10]
    twin.attrs["t_brake"] = float(t0)
    return seg, twin, t_cross


def feats(g, e, ap):
    rw = ap.runways[e.runway]
    x, y, t = g.x.values, g.y.values, g.t.values
    _, cr = rw.along_cross(x, y)
    dperp = np.abs(cr) - e.offset                       # metres left before the protected area
    dt = np.clip(np.diff(t, prepend=t[0] - 5), 0.5, None)
    vc = -np.diff(np.abs(cr), prepend=np.abs(cr[0])) / dt  # closing speed toward the runway (m/s)
    vc = pd.Series(vc).ewm(alpha=0.6).mean().values
    gs = g.gs_obs.values if "gs_obs" in g else g.gs_s.values
    gs = np.where(np.isnan(gs), np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) / dt / KT, gs)
    acc = np.diff(gs, prepend=gs[0]) * KT / dt
    areq = np.clip(vc, 0, None) ** 2 / (2 * np.clip(dperp, 1, None))
    ttc = np.where(vc > 0.3, np.clip(dperp, 0, None) / np.clip(vc, 0.3, None), 120)
    return np.column_stack([dperp, vc, gs, acc, areq, np.clip(ttc, 0, 120), dt])


def brake_norm(TR, ap):
    """Where compliant aircraft start braking: (speed at onset, distance to protected area) from
    the real training stops. Returns f(gs) = the q-quantile onset distance (closest 5 % etc.)."""
    pts = []
    for e, s, f, w, fw, tc in TR:
        tb = w.attrs["t_brake"]
        i = np.argmin(np.abs(s.t.values - tb))
        pts.append((f[i, 2], f[i, 0]))
    pts = np.array(pts)
    pts = pts[(pts[:, 0] > 2) & (pts[:, 1] > 0)]
    # distance ~ c * speed: fit the ratio's quantiles
    ratio = pts[:, 1] / pts[:, 0]
    return lambda gs, q: np.quantile(ratio, q) * gs, pts


def scores(F, model, kind, norm=None):
    """Monotone alert score per fix: alert when score > threshold."""
    dperp, vc, gs, acc, areq = F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4]
    if kind == "geofence":
        return np.where(gs > 3, -dperp, -np.inf)
    if kind == "kinematic":
        return np.where((vc > 0.5) & (dperp < 150), areq, -np.inf)
    if kind == "clearance_norm":
        # with a hold-short clearance known: fraction of the compliant braking-onset distance
        # (at this speed) already used up, while the aircraft is not decelerating
        ref = np.clip(norm(np.clip(gs, 1, None), 0.05), 5, None)  # 95 % of compliant aircraft brake before this
        return np.where((acc > -0.15) & (vc > 0.3), 1 - dperp / ref, -np.inf)
    return model.predict_proba(F)[:, 1]


def main():
    ap = Airport()
    d = pd.read_parquet(ROOT / "data/tracks.parquet")
    E = find_events(d, ap)
    G = {k: g for k, g in d.groupby("track_id")}
    t_split = TEST_START
    pairs = []
    for _, e in E.iterrows():
        seg, twin, tc = approach(G[e.track_id], e, ap)
        if twin is not None:
            pairs.append((e, seg, twin, tc))
    tr = [p for p in pairs if p[0].t_stop < t_split]; te = [p for p in pairs if p[0].t_stop >= t_split]
    print(E.groupby("sensor").size().to_dict(), flush=True)
    print(f"{len(E)} real hold-short stops; {len(pairs)} with a valid no-stop twin; train {len(tr)}, test {len(te)}", flush=True)
    hours_test = (d.t.max() - t_split) / 3600

    def camera_feed(g, rng, rate=2.0, sig=4.0):
        """Camera-only input (cooperative surveillance lost): positions at 2 Hz, 4 m noise, no speed."""
        tt = np.arange(g.t.iloc[0], g.t.iloc[-1] + 1e-6, 1 / rate)
        out = pd.DataFrame({"t": tt, "x": np.interp(tt, g.t, g.x) + rng.normal(0, sig, len(tt)),
                            "y": np.interp(tt, g.t, g.y) + rng.normal(0, sig, len(tt))})
        # smooth like a tracker would before differencing (EWMA on positions)
        out[["x", "y"]] = out[["x", "y"]].ewm(alpha=0.35).mean()
        out["gs_obs"] = np.nan; out["level"] = 4; out["ground"] = True
        return out

    def prep(ps, level, seed):
        rng = np.random.default_rng(seed); out = []
        for e, seg, twin, tc in ps:
            if level == 4:
                s = camera_feed(seg, rng); w = camera_feed(twin, rng)
                w = w[w.t <= tc].copy(); w.attrs["t_brake"] = twin.attrs["t_brake"]; s.attrs["t_brake"] = twin.attrs["t_brake"]
                if len(w) < 2: continue
                out.append((e, s, feats(s, e, ap), w, feats(w, e, ap), tc)); continue
            s = degrade(seg.assign(gs=seg.gs_s), level, rng) if level else seg.assign(gs_obs=seg.gs_s)
            w = degrade(twin.assign(gs=twin.gs_s), level, np.random.default_rng(rng.integers(1 << 30))) if level else twin.assign(gs_obs=twin.gs_s)
            w = w[w.t <= tc].copy()  # an alert after the crossing is no alert
            w.attrs["t_brake"] = twin.attrs["t_brake"]
            s.attrs["t_brake"] = twin.attrs["t_brake"]
            if len(w) < 2:
                continue
            out.append((e, s, feats(s, e, ap), w, feats(w, e, ap), tc))
        return out

    res = {"events": {"real_stops": int(len(E)), "pairs": len(pairs), "train": len(tr), "test": len(te),
                      "test_hours": hours_test, "median_hold_s": float(E.hold_s.median())}}
    for level in [0, 1, 2, 3, 4]:
        TR, TE = prep(tr, level, 50 + level), prep(te, level, 90 + level)
        X = np.vstack([f for _, _, f, _, _, _ in TR] + [f for _, _, _, _, f, _ in TR])
        y = np.concatenate([np.zeros(len(f)) for _, _, f, _, _, _ in TR] +
                           [((tc - w.t.values) <= HORIZON).astype(float) for _, _, _, w, _, tc in TR])
        mk = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                    min_samples_leaf=20, random_state=0)
        gbm = mk().fit(X, y)
        # out-of-fold scores for the TRAINING events (5 folds by event), so the alert threshold
        # is chosen on predictions the model did not train on; in-sample scores look too clean
        nE = len(TR); fold = np.random.default_rng(0).permutation(nE) % 5
        oof_stop, oof_twin = [None] * nE, [None] * nE
        for k in range(5):
            tr_i = [i for i in range(nE) if fold[i] != k]
            Xk = np.vstack([TR[i][2] for i in tr_i] + [TR[i][4] for i in tr_i])
            yk = np.concatenate([np.zeros(len(TR[i][2])) for i in tr_i] +
                                [((TR[i][5] - TR[i][3].t.values) <= HORIZON).astype(float) for i in tr_i])
            mdl = mk().fit(Xk, yk)
            for i in range(nE):
                if fold[i] == k:
                    oof_stop[i] = mdl.predict_proba(TR[i][2])[:, 1]; oof_twin[i] = mdl.predict_proba(TR[i][4])[:, 1]
        res[level] = {}
        norm, onset = brake_norm(TR, ap)
        grids = {"geofence": np.linspace(-150, -5, 30), "kinematic": np.geomspace(0.05, 3, 30),
                 "gbm": np.linspace(0.05, 0.95, 37), "clearance_norm": np.linspace(-1, 1, 41)}
        for kind, grid in grids.items():
            sc_tr = oof_stop if kind == "gbm" else [scores(f, gbm, kind, norm) for _, _, f, _, _, _ in TR]
            sc_te = [scores(f, gbm, kind, norm) for _, _, f, _, _, _ in TE]
            sc_tw = [scores(f, gbm, kind, norm) for _, _, _, _, f, _ in TE]
            mx_tr = np.array([x.max() if len(x) else -np.inf for x in sc_tr]); mx_te = np.array([x.max() if len(x) else -np.inf for x in sc_te])
            avail = np.array([tc - w.attrs["t_brake"] for _, _, _, w, _, tc in TE])
            curve = []
            for thr in grid:
                fa_tr = float(np.mean(mx_tr > thr)); fa = float(np.mean(mx_te > thr))
                leads = np.array([(tc - w.t.values[x > thr][0]) if (x > thr).any() else 0.0
                                  for (_, _, _, w, _, tc), x in zip(TE, sc_tw)])
                curve.append({"thr": float(thr), "false_alert_rate_train": fa_tr, "false_alert_rate": fa,
                              "false_alerts_per_hour": float(fa * len(TE) / hours_test),
                              "detected": float(np.mean(leads > 0)), "lead_median_s": float(np.median(leads)),
                              "lead_p10_s": float(np.percentile(leads, 10)),
                              "lead_ge_10s": float(np.mean(leads >= 10)),
                              "available_median_s": float(np.median(avail)),
                              "lead_fraction_of_available": float(np.median(leads / np.clip(avail, 1e-6, None)))})
            # operating point chosen on TRAINING events: lowest threshold with <= 5 % false alerts on train stops
            ok = [c for c in curve if c["false_alert_rate_train"] <= 0.05]
            op = max(ok, key=lambda c: c["lead_median_s"]) if ok else min(curve, key=lambda c: c["false_alert_rate_train"])
            res[level][kind] = {"curve": curve, "operating_point": op}
            if kind == "clearance_norm":
                res[level]["brake_onset_s_per_kt_q05_q50"] = [float(norm(1.0, 0.05)), float(norm(1.0, 0.5))]
            print(f"level {level} {kind:9s}: thr {op['thr']:.2f}  FA test {op['false_alert_rate']:.3f} ({op['false_alerts_per_hour']:.2f}/h)"
                  f"  detected {op['detected']:.2f}  lead median {op['lead_median_s']:.1f}s of {op['available_median_s']:.1f}s available  p10 {op['lead_p10_s']:.1f}s  >=10s {op['lead_ge_10s']:.2f}", flush=True)
    json.dump(res, open(ROOT / "reports/incursion.json", "w"), indent=1)


if __name__ == "__main__":
    main()

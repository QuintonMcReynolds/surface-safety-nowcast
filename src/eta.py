"""Time-to-event forecasting for departures with calibrated uncertainty.

At every surveillance fix while a departure taxis, predict (a) seconds until it enters the
runway and (b) seconds until its takeoff roll starts. The assigned runway end is treated as
known (it is in the taxi clearance); the entry intersection, queue and timing are not.

Models: physics baseline (route distance / typical taxi speed + median queue wait), quantile
gradient boosting (10/50/90 %), and conformalised quantile regression (CQR, Romano et al.
2019) on held-out calibration tracks, which restores the 80 % coverage the raw quantiles miss.
Split by time: tracks rolling in the first 4 h train/calibrate, the last 2 h test."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
from airport import Airport
from activity import degrade
from tracks import S, TEST_START

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.2  # 80 % intervals


def runway_end_xy(ap, rwy, roll_xy):
    """Threshold of the runway end the aircraft departed from (nearest end to the roll start)."""
    r = ap.runways[rwy]
    return r.a if np.linalg.norm(roll_xy - r.a) < np.linalg.norm(roll_xy - r.b) else r.b


def build_rows(d, ev, ap, level, seed):
    rng = np.random.default_rng(seed)
    allfix = d[["t", "x", "y", "gs_s", "state", "track_id"]]
    rows = []
    deps = ev[ev.roll_start.notna() & (ev.sensor == "adsb") & ~ev.vehicle]
    roll_times = deps.set_index("dep_rwy").roll_start.sort_values()
    arr_t = ev.touchdown.dropna().values; all_roll = deps.roll_start.values
    for _, e in deps.iterrows():
        g = d[d.track_id == e.track_id]
        g0 = g[g.t <= e.roll_start]
        roll_xy = g[g.t >= e.roll_start][["x", "y"]].values[0]
        end = runway_end_xy(ap, e.dep_rwy, roll_xy)
        holds = [h for h in ap.holds if h.runway == e.dep_rwy and np.linalg.norm(h.xy - end) < 900]
        target = min(holds, key=lambda h: np.linalg.norm(h.xy - end)).xy if holds else end
        moving = np.where(g0.gs_s.values > 3)[0]
        if not len(moving):
            continue
        g0 = g0.iloc[moving[0]:]
        g0 = g0[g0.t >= e.roll_start - 1800]
        gd = degrade(g0.assign(gs=g0.gs), level, rng)
        same_rwy = roll_times.get(e.dep_rwy)
        same_rwy = np.atleast_1d(same_rwy.values if hasattr(same_rwy, "values") else same_rwy)
        t_move = gd.t.iloc[0]
        gs_hist = []
        for _, f in gd.iterrows():
            if f.t >= e.dep_rwy_enter:  # predictions stop once on the runway
                break
            gs_hist.append((f.t, f.gs_obs if not np.isnan(f.gs_obs) else np.nan))
            recent = [v for tt, v in gs_hist if tt > f.t - 60 and not np.isnan(v)]
            # queue: other targets stopped/slow within 450 m of this runway end, right now (any sensor)
            now = allfix[(allfix.t > f.t - 6) & (allfix.t <= f.t) & (allfix.track_id != e.track_id)]
            near = np.hypot(now.x - target[0], now.y - target[1]) < 450
            queue = int(((now.gs_s < 5) & near).sum())
            dist_t = np.hypot(now.x - target[0], now.y - target[1])
            inbound = int(((now.gs_s >= 3) & (dist_t < 1500) & (dist_t >= 450)).sum())
            rows.append({
                "track_id": e.track_id, "t": f.t, "level": level,
                "route_m": ap.route_distance(f.x, f.y, target),
                "direct_m": float(np.hypot(f.x - target[0], f.y - target[1])),
                "gs": f.gs_obs, "gs_60s": np.mean(recent) if recent else np.nan,
                "since_move_s": f.t - t_move, "queue": queue,
                "deps_same_rwy_10min": int(((same_rwy > f.t - 600) & (same_rwy <= f.t)).sum()),
                "inbound_1500m": inbound,
                "arrivals_10min": int(((arr_t > f.t - 600) & (arr_t <= f.t)).sum()),
                "deps_all_10min": int(((all_roll > f.t - 600) & (all_roll <= f.t)).sum()),
                "hour": (f.t % 86400) / 3600,
                "rwy": e.dep_rwy,
                "y_enter": e.dep_rwy_enter - f.t, "y_roll": e.roll_start - f.t,
                "roll_start": e.roll_start,
            })
    return pd.DataFrame(rows)


FEATS = ["route_m", "direct_m", "gs", "gs_60s", "since_move_s", "queue", "inbound_1500m",
         "deps_same_rwy_10min", "deps_all_10min", "arrivals_10min"]


def fit_q(X, y, q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=40, random_state=0).fit(X, y)


def evaluate(te, lo, mid, hi, ytrue):
    lo, mid, hi, ytrue = (np.asarray(v, float) for v in (lo, mid, hi, ytrue))
    cover = (ytrue >= lo) & (ytrue <= hi)
    out = {"n": int(len(ytrue)), "tracks": int(te.track_id.nunique()),
           "mae_median": float(np.mean(np.abs(mid - ytrue))), "coverage80": float(cover.mean()),
           "mean_width": float(np.mean(hi - lo))}
    # by horizon
    bins = [0, 60, 180, 420, 1e9]; lab = ["<1 min", "1-3 min", "3-7 min", ">7 min"]
    h = pd.cut(ytrue, bins, labels=lab)
    out["by_horizon"] = {str(k): {"n": int((h == k).sum()), "mae": float(np.mean(np.abs(mid - ytrue)[h == k])),
                                  "coverage80": float(cover[h == k].mean()), "width": float(np.mean((hi - lo)[h == k]))}
                         for k in lab if (h == k).sum() > 10}
    # bootstrap over tracks for coverage CI
    ids = te.track_id.values; u = np.unique(ids); rng = np.random.default_rng(0)
    covs = []
    for _ in range(300):
        pick = rng.choice(u, len(u))
        m = np.concatenate([np.where(ids == p)[0] for p in pick])
        covs.append(cover[m].mean())
    out["coverage80_ci95"] = [float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))]
    return out


def aci(te, lo, hi, y, cal_scores, alpha=ALPHA, gamma=0.02):
    """Adaptive conformal inference (Gibbs & Candes 2021). Predictions are issued in time order;
    a prediction's outcome is only learned when its aircraft starts the roll, so alpha_t is
    updated from outcomes that have resolved by the time each prediction is made."""
    order = np.argsort(te.t.values)
    t_issue, t_resolve = te.t.values, te.roll_start.values
    scores = list(cal_scores)
    a_t = alpha
    lo_o, hi_o = np.zeros(len(y)), np.zeros(len(y))
    pending = []  # (resolve_time, idx)
    import heapq
    for i in order:
        while pending and pending[0][0] <= t_issue[i]:
            _, j, q_j = heapq.heappop(pending)
            err = float(not (lo[j] - q_j <= y[j] <= hi[j] + q_j))
            a_t = a_t + gamma * (alpha - err)
            scores.append(max(lo[j] - y[j], y[j] - hi[j]))
        lvl = np.clip(1 - a_t, 0, 1)
        q = np.quantile(scores[-2000:], lvl) if lvl < 1 else max(scores) * 1.5
        lo_o[i], hi_o[i] = max(lo[i] - q, 0), hi[i] + q
        heapq.heappush(pending, (t_resolve[i], i, q))
    return lo_o, hi_o


def main():
    ap = Airport()
    d = pd.read_parquet(ROOT / "data/tracks.parquet")
    ev = pd.read_parquet(ROOT / "data/events.parquet")
    t0 = d.t.min()
    res = {}
    for level in [0, 2]:
        R = build_rows(d, ev, ap, level, seed=7 + level).dropna(subset=["route_m"])
        R["hr"] = np.floor((R.roll_start - TEST_START) / 3600).astype(int)  # <0 = training days
        if level == 0:
            R.to_parquet(ROOT / "data/eta_rows_level0.parquet")
        res[level] = {}
        for tgt in ["y_enter", "y_roll"]:
            out = {k: {"lo": [], "mid": [], "hi": [], "y": [], "rows": []} for k in ["physics", "quantile_gbm", "cqr_static", "cqr_aci"]}
            for h in range(6):  # rolling origin over the test day: train on every departure before hour h
                past = R[R.roll_start < TEST_START + h * 3600]; te = R[R.hr == h]
                if not len(te): continue
                cut = past.roll_start.quantile(0.75)
                fit, cal = past[past.roll_start <= cut], past[past.roll_start > cut]
                speed = 15 * 0.514444
                wait = np.median(fit[tgt] - fit.route_m / speed)
                resid = past[tgt] - (past.route_m / speed + wait)
                ql, qh = np.quantile(resid, [ALPHA / 2, 1 - ALPHA / 2])
                b = te.route_m.values / speed + wait
                ms = {q: fit_q(fit[FEATS], fit[tgt], q) for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2)}
                lo, mid, hi = (ms[q].predict(te[FEATS]) for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2))
                clo, chi = ms[ALPHA / 2].predict(cal[FEATS]), ms[1 - ALPHA / 2].predict(cal[FEATS])
                sc = np.maximum(clo - cal[tgt].values, cal[tgt].values - chi)
                qhat = np.quantile(sc, min(1, (1 - ALPHA) * (1 + 1 / len(sc))))
                alo, ahi = aci(te, lo, hi, te[tgt].values, sc)
                for k, (l_, m_, h_) in {"physics": (b + ql, b, b + qh), "quantile_gbm": (lo, mid, hi),
                                        "cqr_static": (np.maximum(lo - qhat, 0), mid, hi + qhat),
                                        "cqr_aci": (alo, mid, ahi)}.items():
                    out[k]["lo"].append(l_); out[k]["mid"].append(m_); out[k]["hi"].append(h_)
                    out[k]["y"].append(te[tgt].values); out[k]["rows"].append(te)
            r = {}
            for k, v in out.items():
                te_all = pd.concat(v["rows"])
                r[k] = evaluate(te_all, *(np.concatenate(v[c]) for c in ["lo", "mid", "hi", "y"]))
                r[k]["coverage80_by_test_hour"] = {int(hh): float(np.mean((yy >= ll) & (yy <= hi_)))
                                                   for hh, ll, hi_, yy in zip(range(6), v["lo"], v["hi"], v["y"])}
            res[level][tgt] = r
            print(f"level {level} {tgt}: " + "  ".join(f"{k} MAE {x['mae_median']:.0f}s cov {x['coverage80']:.2f} {x['coverage80_by_test_hour']} width {x['mean_width']:.0f}s"
                                                       for k, x in r.items()), flush=True)
        if level == 0:
            past = R[R.hr < 0]; te = R[R.hr >= 0]
            m = fit_q(past[FEATS], past.y_roll, 0.5)
            base = np.mean(np.abs(m.predict(te[FEATS]) - te.y_roll)); rng = np.random.default_rng(0)
            imp = {}
            for f in FEATS:
                X = te[FEATS].copy(); X[f] = rng.permutation(X[f].values)
                imp[f] = float(np.mean(np.abs(m.predict(X) - te.y_roll)) - base)
            res["perm_importance_roll_mae_s"] = imp
            print("importance", {k: round(v) for k, v in imp.items()}, flush=True)
    json.dump(res, open(ROOT / "reports/eta.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Multi-sensor surface tracking: real ADS-B + real TIS-B ground radar + a simulated camera.

Camera (simulated, because no labelled KIAD surface video is public): fixed at a busy runway
hold area, 2 Hz, detects every target in a 450 m radius with Pd = 0.9, position noise
2 m + 1 % of range, Poisson clutter, no identities, and a clock running 0.4 s late.
Truth: each real track smoothed and interpolated to 0.5 s; the camera samples that truth.

Trackers (all causal):
  coop_only  EKF per target on its own ADS-B / TIS-B reports (identity known); state
             [x, y, vx, vy]; ADS-B ground speed + track make the update nonlinear (EKF).
  fused      same EKF plus camera detections assigned by JPDA (exact joint-event enumeration
             within gate clusters), camera clock offset estimated from the first 20 minutes.
  camera_only  JPDA-tracked from camera detections alone, initiated M-of-N, for the case
             where the cooperative input is lost (transponder off, ground station down).
Metrics: position RMSE vs truth, share of time with error > 15 m (the margin that decides
which side of a hold line a target is on), association accuracy, clock-offset estimate."""
import itertools, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).parent))
from airport import Airport
from tracks import TEST_START

ROOT = Path(__file__).resolve().parents[1]
KT = 0.514444
CAM = dict(rate=2.0, radius=450.0, pd=0.9, sig0=2.0, sig_r=0.01, clutter=1.0, offset=0.4)
GATE = chi2.ppf(0.99, 2)
Q_ACC = 0.6  # m/s^2 white-acceleration intensity (taxi manoeuvres)


def truth_paths(d, centre, t_lo, t_hi):
    out = {}
    for tid, g in d[(d.t >= t_lo) & (d.t <= t_hi) & (d.ground_flag)].groupby("track_id"):
        if len(g) < 6: continue
        dist = np.hypot(g.x - centre[0], g.y - centre[1])
        if dist.min() > CAM["radius"]: continue
        sx = g.x.rolling(3, center=True, min_periods=1).mean().values
        sy = g.y.rolling(3, center=True, min_periods=1).mean().values
        tt = np.arange(g.t.iloc[0], g.t.iloc[-1], 0.5)
        out[tid] = pd.DataFrame({"t": tt, "x": np.interp(tt, g.t, sx), "y": np.interp(tt, g.t, sy)})
    return out


def camera(truth, centre, t_lo, t_hi, rng):
    frames = []
    for tf in np.arange(t_lo, t_hi, 1 / CAM["rate"]):
        det = []
        for tid, p in truth.items():
            if tf < p.t.iloc[0] or tf > p.t.iloc[-1]: continue
            x, y = np.interp(tf, p.t, p.x), np.interp(tf, p.t, p.y)
            r = np.hypot(x - centre[0], y - centre[1])
            if r > CAM["radius"] or rng.random() > CAM["pd"]: continue
            s = CAM["sig0"] + CAM["sig_r"] * r
            det.append((x + rng.normal(0, s), y + rng.normal(0, s), s, tid))
        for _ in range(rng.poisson(CAM["clutter"])):
            a, rr = rng.uniform(0, 2 * np.pi), CAM["radius"] * np.sqrt(rng.random())
            det.append((centre[0] + rr * np.cos(a), centre[1] + rr * np.sin(a), 10.0, None))
        frames.append((tf + CAM["offset"], det))  # stamped by a clock running late
    return frames


class EKF:
    def __init__(self, t, x, y, vx=0.0, vy=0.0, P=None):
        self.t, self.s = t, np.array([x, y, vx, vy], float)
        self.P = P if P is not None else np.diag([25.0, 25.0, 25.0, 25.0])
        self.hits, self.miss, self.confirmed = 1, 0, False

    def predict(self, t):
        dt = max(t - self.t, 0.0)
        if dt == 0: return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.s = F @ self.s; self.P = F @ self.P @ F.T + G @ G.T * Q_ACC ** 2; self.t = t

    def update_pos(self, z, R):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.]])
        S = H @ self.P @ H.T + R; K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ (z - H @ self.s); self.P = (np.eye(4) - K @ H) @ self.P

    def update_adsb(self, z, R):
        """z = [x, y, gs (m/s), track (rad)] - nonlinear in velocity: linearise (EKF)."""
        x, y, vx, vy = self.s
        sp = max(np.hypot(vx, vy), 0.3)
        h = np.array([x, y, sp, np.arctan2(vx, vy)])
        H = np.zeros((4, 4)); H[0, 0] = H[1, 1] = 1
        H[2, 2], H[2, 3] = vx / sp, vy / sp
        H[3, 2], H[3, 3] = vy / sp ** 2, -vx / sp ** 2
        r = z - h; r[3] = (r[3] + np.pi) % (2 * np.pi) - np.pi
        if z[2] < 1.0:  # heading meaningless when (nearly) stopped
            H, r, R = H[:3], r[:3], R[:3, :3]
        S = H @ self.P @ H.T + R; K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ r; self.P = (np.eye(4) - K @ H) @ self.P

    def innov(self, z, R):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.]])
        S = H @ self.P @ H.T + R; v = z - H @ self.s
        return v, S, float(v @ np.linalg.solve(S, v))


def jpda(tracks, dets, pd_=CAM["pd"], lam=None):
    """Exact JPDA weights beta[track][j] (j = -1 means no detection) for one frame.
    Returns the weights and, per detection, the most probable source track."""
    lam = lam or CAM["clutter"] / (np.pi * CAM["radius"] ** 2)
    L = {}
    for ti, tr in enumerate(tracks):
        for j, (x, y, s, _) in enumerate(dets):
            v, S, d2 = tr.innov(np.array([x, y]), np.eye(2) * s ** 2)
            if d2 < GATE:
                L[ti, j] = pd_ * np.exp(-0.5 * d2) / (2 * np.pi * np.sqrt(np.linalg.det(S))) / lam
    beta = {ti: {-1: 0.0} for ti in range(len(tracks))}
    # clusters of tracks connected through shared gated detections
    parent = list(range(len(tracks)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    by_det = {}
    for ti, j in L: by_det.setdefault(j, []).append(ti)
    for ts in by_det.values():
        for a in ts[1:]: parent[find(a)] = find(ts[0])
    clusters = {}
    for ti in range(len(tracks)): clusters.setdefault(find(ti), []).append(ti)
    for cl in clusters.values():
        opts = [[-1] + [j for (t2, j) in L if t2 == ti] for ti in cl]
        if np.prod([len(o) for o in opts]) > 500:  # huge cluster: independent PDA per track
            for ti, o in zip(cl, opts):
                w = {j: (L[ti, j] if j >= 0 else (1 - pd_)) for j in o}
                tot = sum(w.values()); beta[ti] = {j: v / tot for j, v in w.items()}
            continue
        tot = 0.0; acc = {ti: {} for ti in cl}
        for combo in itertools.product(*opts):
            used = [j for j in combo if j >= 0]
            if len(used) != len(set(used)): continue
            w = 1.0
            for ti, j in zip(cl, combo):
                w *= L[ti, j] if j >= 0 else (1 - pd_)
            tot += w
            for ti, j in zip(cl, combo): acc[ti][j] = acc[ti].get(j, 0.0) + w
        for ti in cl:
            beta[ti] = {j: w / tot for j, w in acc[ti].items()}
    return beta


def pda_update(tr, beta, dets):
    """Probabilistic data association update with the spread-of-innovations term."""
    if not beta or set(beta) == {-1}: return
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.]])
    sig = np.mean([dets[j][2] for j in beta if j >= 0]); R = np.eye(2) * sig ** 2
    S = H @ tr.P @ H.T + R; K = tr.P @ H.T @ np.linalg.inv(S)
    vs = {j: np.array(dets[j][:2]) - H @ tr.s for j in beta if j >= 0}
    v = sum(beta[j] * vs[j] for j in vs)
    b0 = beta.get(-1, 0.0)
    Pc = tr.P - K @ S @ K.T
    spread = K @ (sum(beta[j] * np.outer(vs[j], vs[j]) for j in vs) - np.outer(v, v)) @ K.T
    tr.s = tr.s + K @ v; tr.P = b0 * tr.P + (1 - b0) * Pc + spread


def coop_reports(d, tids, t_lo, t_hi):
    r = d[d.track_id.isin(tids) & (d.t >= t_lo) & (d.t <= t_hi)][["t", "track_id", "x", "y", "gs", "track", "nac_p", "sensor"]]
    return r.sort_values("t")


def run(d, truth, frames, t_lo, t_hi, mode, offset_hat=0.0, drop=frozenset()):
    """mode: coop_only | fused | camera_only. drop: track ids whose cooperative reports are lost."""
    rep = coop_reports(d, list(truth), t_lo, t_hi) if mode != "camera_only" else pd.DataFrame(columns=["t"])
    rep = rep[~rep.track_id.isin(drop)] if len(rep) else rep
    events = [(r.t, 0, r) for r in rep.itertuples()] + ([(tf - offset_hat, 1, det) for tf, det in frames] if mode != "coop_only" else [])
    events.sort(key=lambda e: (e[0], e[1]))
    trk, est = {}, []  # trk: id -> EKF ; id = track_id for coop tracks, "cam#n" for camera-initiated
    tentative, n_cam, assoc = [], 0, []
    grid = np.arange(np.ceil(t_lo), t_hi, 1.0); gi = 0
    for t, kind, payload in events:
        while gi < len(grid) and grid[gi] <= t:
            for k, tr in trk.items():
                if str(k).startswith("cam#") and not tr.confirmed: continue
                tr2 = EKF(tr.t, *tr.s, P=tr.P.copy()); tr2.predict(grid[gi]); est.append((grid[gi], k, tr2.s[0], tr2.s[1]))
            gi += 1
        if kind == 0:
            r = payload
            sig = 3.0 if r.sensor == "adsb" and (r.nac_p or 0) >= 9 else 10.0 if r.sensor == "adsb" else 15.0
            if r.track_id not in trk:
                trk[r.track_id] = EKF(t, r.x, r.y); continue
            tr = trk[r.track_id]; tr.predict(t)
            if r.sensor == "adsb" and not np.isnan(r.gs) and not np.isnan(r.track):
                tr.update_adsb(np.array([r.x, r.y, r.gs * KT, np.radians(r.track)]), np.diag([sig ** 2, sig ** 2, 0.5 ** 2, np.radians(8) ** 2]))
            else:
                tr.update_pos(np.array([r.x, r.y]), np.eye(2) * sig ** 2)
        else:
            dets = payload
            for tr in trk.values(): tr.predict(t)
            # drop stale tracks (no update for 30 s)
            keys = list(trk)
            beta = jpda([trk[k] for k in keys], dets)
            claimed = set()
            for i, k in enumerate(keys):
                b = beta[i]; pda_update(trk[k], b, dets)
                if b and max(b, key=b.get) >= 0:
                    j = max(b, key=b.get); claimed.add(j)
                    assoc.append((dets[j][3], k))
                    trk[k].hits += 1; trk[k].miss = 0
                    if trk[k].hits >= 4: trk[k].confirmed = True
                else:
                    trk[k].miss += 1
            if mode != "coop_only":  # camera-initiated tracks for targets nobody else reports
                for k in [k for k in keys if str(k).startswith("cam#") and trk[k].miss > 6]: del trk[k]
                for j, (x, y, s, src) in enumerate(dets):
                    if j not in claimed and not any(np.hypot(x - trk[k].s[0], y - trk[k].s[1]) < 20 for k in trk):
                        n_cam += 1; trk[f"cam#{n_cam}"] = EKF(t, x, y, P=np.diag([s ** 2, s ** 2, 16.0, 16.0]))
    return pd.DataFrame(est, columns=["t", "id", "x", "y"]), assoc


def score(est, truth, centre, cam_ids=False):
    """Error vs truth on a 1 Hz grid, inside the camera area. For camera-only, each truth
    target is matched to the nearest confirmed track each second (gaps count as misses)."""
    errs, covered, total = [], 0, 0
    est = est.assign(t=np.round(est.t).astype(int))
    E = {k: g.drop_duplicates("t").set_index("t") for k, g in est.groupby("id")}
    for tid, p in truth.items():
        tt = np.arange(np.ceil(p.t.iloc[0]), p.t.iloc[-1], 1.0).astype(int)
        x, y = np.interp(tt, p.t, p.x), np.interp(tt, p.t, p.y)
        inside = np.hypot(x - centre[0], y - centre[1]) < CAM["radius"]
        for t_, x_, y_ in zip(tt[inside], x[inside], y[inside]):
            total += 1
            if cam_ids:
                sub = est[est.t == t_]
                if not len(sub): continue
                e = np.hypot(sub.x - x_, sub.y - y_).min()
                if e < 40: covered += 1; errs.append(e)
            elif tid in E and t_ in E[tid].index:
                r = E[tid].loc[t_]; covered += 1; errs.append(float(np.hypot(r.x - x_, r.y - y_)))
    errs = np.array(errs)
    return {"rmse_m": float(np.sqrt(np.mean(errs ** 2))) if len(errs) else None,
            "p95_m": float(np.percentile(errs, 95)) if len(errs) else None,
            "err_gt_15m": float(np.mean(errs > 15)) if len(errs) else None,
            "coverage": covered / max(total, 1), "target_seconds": total}


def main():
    ap = Airport()
    d = pd.read_parquet(ROOT / "data/tracks.parquet")
    d["ground_flag"] = d.ground
    # camera at the hold line with the most real hold-short stops
    hs = d[d.state == 2].hold_idx.value_counts()
    centre = ap.holds[int(hs.index[0])].xy
    t_split = TEST_START
    rng = np.random.default_rng(3)
    # 1) estimate the camera clock offset on a training-day hour: for each candidate offset,
    #    median distance from camera detections to the nearest cooperative report interpolated
    #    to the corrected detection time. Clutter adds a constant floor; the minimum is sharp.
    prev = d[d.t < t_split - 6 * 3600].t.max()  # end of the previous sample day (skip stale pre-noon fixes)
    cal_lo, cal_hi = prev - 3600, prev
    tr_cal = truth_paths(d, centre, cal_lo, cal_hi); fr_cal = camera(tr_cal, centre, cal_lo, cal_hi, rng)
    rep = coop_reports(d, list(tr_cal), cal_lo, cal_hi)
    paths = {k: g for k, g in rep.groupby("track_id") if len(g) > 2}
    cand = np.round(np.arange(-1.0, 1.51, 0.1), 2); fit = []
    # only moving targets carry timing information; parked aircraft and clutter do not
    vel = {k: (np.gradient(g.x.values, g.t.values), np.gradient(g.y.values, g.t.values)) for k, g in paths.items()}
    for off in cand:
        sq = []
        for tf, det in fr_cal[::4]:
            t = tf - off
            P, V = [], []
            for k, g in paths.items():
                if g.t.iloc[0] <= t <= g.t.iloc[-1]:
                    P.append((np.interp(t, g.t, g.x), np.interp(t, g.t, g.y)))
                    V.append(np.hypot(np.interp(t, g.t, vel[k][0]), np.interp(t, g.t, vel[k][1])))
            if not P: continue
            P, V = np.array(P), np.array(V)
            for x, y, s_, _ in det:
                dd = np.hypot(P[:, 0] - x, P[:, 1] - y); j = int(np.argmin(dd))
                if V[j] > 3.0 and dd[j] < 40:
                    sq.append(dd[j] ** 2)
        fit.append(-np.mean(sq) if sq else -np.inf)
    off_hat = float(cand[int(np.argmax(fit))])
    print(f"camera clock offset: true {CAM['offset']:.2f}s, estimated {off_hat:.2f}s (calibration hour on a training day)", flush=True)
    res_cal = {"offset_grid": cand.tolist(), "neg_median_dist_m": [float(v) for v in fit]}
    # 2) evaluate on the test window
    lo, hi = t_split, t_split + 1800  # first 30 min of the test day
    truth = truth_paths(d, centre, lo, hi); frames = camera(truth, centre, lo, hi, rng)
    sens = d[d.track_id.isin(truth)].groupby("track_id").sensor.first()
    print(f"test window (30 min): {len(truth)} targets through the camera area ({sens.value_counts().to_dict()}), {len(frames)} frames", flush=True)
    res = {"camera": CAM, "offset_estimate_s": off_hat, "offset_calibration": res_cal, "targets": int(len(truth)), "by_sensor": sens.value_counts().to_dict()}
    for name, kw in {"coop_only": dict(mode="coop_only"),
                     "fused_no_clock_fix": dict(mode="fused", offset_hat=0.0),
                     "fused": dict(mode="fused", offset_hat=off_hat)}.items():
        est, assoc = run(d, truth, frames, lo, hi, **kw)
        r = score(est, truth, centre)
        if assoc:
            a = [(src, k) for src, k in assoc]
            r["assoc_correct"] = float(np.mean([src == k for src, k in a if src is not None]))
            r["clutter_assigned"] = float(np.mean([src is None for src, k in a]))
        res[name] = r
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    # 3) cooperative input lost for every target: camera-only multi-target tracking
    est, assoc = run(d, truth, frames, lo, hi, "camera_only")
    res["camera_only_coop_lost"] = score(est, truth, centre, cam_ids=True)
    est_c, _ = run(d, truth, frames, lo, hi, "coop_only")
    res["coop_only_matched_scoring"] = score(est_c, truth, centre)
    print("camera_only (coop lost)", res["camera_only_coop_lost"], flush=True)
    # 4) one-at-a-time: each ADS-B aircraft loses its transponder; fused tracker keeps it on camera
    adsb_ids = [t for t in truth if sens.get(t) == "adsb"]
    est, _ = run(d, truth, frames, lo, hi, "fused", offset_hat=off_hat, drop=frozenset(adsb_ids[::2]))
    sub = {k: v for k, v in truth.items() if k in adsb_ids[::2]}
    res["fused_half_adsb_dropped_on_dropped"] = score(est, sub, centre, cam_ids=True)
    est, _ = run(d, truth, frames, lo, hi, "coop_only", drop=frozenset(adsb_ids[::2]))
    res["coop_only_half_adsb_dropped_on_dropped"] = score(est, sub, centre, cam_ids=True)
    print("dropped targets: fused", res["fused_half_adsb_dropped_on_dropped"], " coop-only", res["coop_only_half_adsb_dropped_on_dropped"], flush=True)
    json.dump(res, open(ROOT / "reports/fusion.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

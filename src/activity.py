"""Online aircraft-activity inference from noisy, sparse, irregular surface surveillance.

Continuous-time HMM: transitions P(dt) = expm(Q dt) (Q estimated from labelled dwell times,
so irregular sampling is handled exactly), Gaussian-mixture emissions on kinematic and
geometric features, forward filtering only (no look-ahead). Graded against the hindsight
labels from tracks.py at four surveillance-degradation levels.

Calibration fix: emissions trained on clean data become over-confident when inputs degrade.
Each emission log-likelihood is tempered by beta(level) < 1, fitted on validation tracks to
minimise log loss; the level is known online from the reported accuracy (NACp) and update gap."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent))
from airport import Airport
from tracks import STATES, S, TEST_START

ROOT = Path(__file__).resolve().parents[1]
K = len(STATES)
LEVELS = {  # keep-probability, position sigma (m), speed sigma (kt), speed reported?, outage length (s)
    0: dict(keep=1.0, sig=0.0, gs_sig=0.0, gs=True, outage=0),
    1: dict(keep=0.5, sig=10.0, gs_sig=2.0, gs=True, outage=0),
    2: dict(keep=0.25, sig=25.0, gs_sig=4.0, gs=True, outage=30),
    3: dict(keep=0.15, sig=50.0, gs_sig=0.0, gs=False, outage=60),
}


def degrade(g, level, rng):
    c = LEVELS[level]
    g = g.copy()
    keep = rng.random(len(g)) < c["keep"]
    keep[0] = True
    if c["outage"]:
        for _ in range(max(1, int((g.t.iloc[-1] - g.t.iloc[0]) / 600))):  # one outage per ~10 min of track
            t0 = rng.uniform(g.t.iloc[0], g.t.iloc[-1])
            keep &= ~((g.t.values > t0) & (g.t.values < t0 + c["outage"]))
        keep[0] = True
    g = g[keep].copy()
    g["x"] += rng.normal(0, c["sig"], len(g)) if c["sig"] else 0
    g["y"] += rng.normal(0, c["sig"], len(g)) if c["sig"] else 0
    if c["gs"]:
        g["gs_obs"] = np.clip(g.gs.fillna(0).values + rng.normal(0, c["gs_sig"], len(g)), 0, None)
    else:
        g["gs_obs"] = np.nan
    g["level"] = level
    return g


def features(g, ap):
    """Online features: each row uses only the current and previous observation."""
    x, y, t = g.x.values, g.y.values, g.t.values
    dt = np.diff(t, prepend=t[0] - 5.0)
    dt = np.clip(dt, 0.5, None)
    disp = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    v_pos = disp / dt / 0.514444  # kt from displacement
    gs = g.gs_obs.values
    gs = np.where(np.isnan(gs), v_pos, gs)
    if g.level.iloc[0] >= 2:  # noisy displacement speed: shrink toward 0 when the step is within noise
        sig = LEVELS[g.level.iloc[0]]["sig"]
        if np.isnan(g.gs_obs.values).all():
            gs = np.clip(disp - 1.4 * sig, 0, None) / dt / 0.514444
    name, rdist, _ = ap.runway_context(x, y)
    _, hdist = ap.nearest_hold(x, y)
    hdg = np.array([ap.runways[n].heading for n in name])
    mov = np.degrees(np.arctan2(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))) % 360
    trk = np.where(g.track.notna().values & (g.level.values < 3), g.track.fillna(0).values, mov)
    diff = np.abs((trk - hdg + 180) % 360 - 180); align = np.minimum(diff, 180 - diff)
    acc = np.diff(gs, prepend=gs[0]) / dt
    air = (~g.ground.values).astype(float)
    F = np.column_stack([np.log1p(np.clip(gs, 0, 400)), np.log1p(rdist), np.log1p(hdist),
                         np.clip(acc, -6, 6), align / 90.0])
    return F, air, dt


class ActivityHMM:
    def __init__(self, n_comp=3, seed=0):
        self.n_comp, self.seed = n_comp, seed
        self.beta = {0: 1.0}

    def fit(self, seqs):
        """seqs: list of (F, air, dt, labels) from clean (level-0) training tracks."""
        # generator from dwell times / transition counts
        T = np.zeros(K); N = np.zeros((K, K)); start = np.ones(K)
        for F, air, dt, lab in seqs:
            start[lab[0]] += 1
            for a, b, d in zip(lab[:-1], lab[1:], dt[1:]):
                T[a] += d
                if a != b: N[a, b] += 1
        Q = (N + 0.1) / (T[:, None] + 1.0)
        np.fill_diagonal(Q, 0); np.fill_diagonal(Q, -Q.sum(1))
        self.Q, self.pi = Q, start / start.sum()
        Fall = np.vstack([s[0] for s in seqs]); A = np.concatenate([s[1] for s in seqs]); L = np.concatenate([s[3] for s in seqs])
        self.gmm, self.p_air = [], np.zeros(K)
        for k in range(K):
            m = L == k
            self.gmm.append(GaussianMixture(min(self.n_comp, max(1, m.sum() // 50)), covariance_type="full",
                                            reg_covar=1e-2, random_state=self.seed).fit(Fall[m]))
            self.p_air[k] = (A[m].sum() + 1) / (m.sum() + 2)
        return self

    def loglik(self, F, air):
        ll = np.column_stack([g.score_samples(F) for g in self.gmm])
        ll += np.where(air[:, None] > 0.5, np.log(self.p_air), np.log1p(-self.p_air))[: len(F)]
        return ll

    def filter(self, F, air, dt, beta=1.0, ll=None):
        ll = beta * (self.loglik(F, air) if ll is None else ll)
        out = np.zeros((len(F), K))
        logp = np.log(self.pi) + ll[0]; logp -= logsumexp(logp); out[0] = logp
        cache = {}
        for i in range(1, len(F)):
            key = round(float(dt[i]), 1)
            if key not in cache:
                cache[key] = np.log(np.clip(expm(self.Q * key), 1e-12, None))
            logp = logsumexp(logp[:, None] + cache[key], axis=0) + ll[i]
            logp -= logsumexp(logp); out[i] = logp
        return np.exp(out)

    def fit_beta(self, val_by_level, llf=None):
        """Grid-search the emission temperature per degradation level on validation tracks."""
        betas = {}
        for lvl, seqs in val_by_level.items():
            best = None
            for b in np.r_[0.02, 0.04, 0.06, 0.08, np.linspace(0.1, 1.0, 19)]:
                nll = np.mean(np.concatenate([-np.log(np.clip(self.filter(F, a, d, b, None if llf is None else llf(F, a, d))[np.arange(len(l)), l], 1e-9, None))
                                              for F, a, d, l in seqs]))
                if best is None or nll < best[0]: best = (nll, b)
            betas[lvl] = float(best[1])
        return betas


def rule_classifier(F, air):
    """Deterministic threshold logic, the kind of thing a first prototype ships with."""
    gs = np.expm1(F[:, 0]); rd = np.expm1(F[:, 1]); hd = np.expm1(F[:, 2])
    lab = np.full(len(F), S["TAXI"])
    lab[gs < 1.5] = S["STOPPED"]; lab[(gs < 1.5) & (hd < 90) & (rd >= 30)] = S["HOLD_SHORT"]
    on = (rd < 30) & (air < 0.5)
    lab[on] = S["RWY_SLOW"]; lab[on & (gs >= 35) & (F[:, 3] >= 0)] = S["TAKEOFF_ROLL"]
    lab[on & (gs >= 35) & (F[:, 3] < 0)] = S["LANDING_ROLL"]; lab[air > 0.5] = S["AIRBORNE"]
    P = np.full((len(F), K), 0.0); P[np.arange(len(F)), lab] = 1.0
    return P


def metrics(P, y):
    P = np.clip(P, 1e-9, 1); P = P / P.sum(1, keepdims=True)
    pred = P.argmax(1); conf = P.max(1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    f1 = []
    for k in range(K):
        tp = ((pred == k) & (y == k)).sum(); fp = ((pred == k) & (y != k)).sum(); fn = ((pred != k) & (y == k)).sum()
        if (y == k).any(): f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"acc": float((pred == y).mean()), "macro_f1": float(np.mean(f1)),
            "log_loss": float(-np.mean(np.log(P[np.arange(len(y)), y]))), "ece": float(ece),
            "brier": float(np.mean(((P - np.eye(K)[y]) ** 2).sum(1)))}


def reliability(P, y, nb=10):
    conf = P.max(1); hit = P.argmax(1) == y
    bins = np.minimum((conf * nb).astype(int), nb - 1)
    return [(float(conf[bins == b].mean()), float(hit[bins == b].mean()), int((bins == b).sum()))
            for b in range(nb) if (bins == b).sum() > 20]


def main():
    ap = Airport()
    d = pd.read_parquet(ROOT / "data/tracks.parquet")
    d = d[(d.sensor == "adsb") & ~d.vehicle]
    n = d.groupby("track_id").size(); d = d[d.track_id.isin(n[n >= 30].index)]
    t_split = TEST_START
    start = d.groupby("track_id").t.min()
    tr_ids = start[start < t_split].index; te_ids = start[start >= t_split].index
    rng = np.random.default_rng(0)
    tr_ids = rng.permutation(tr_ids); val_ids = set(tr_ids[: len(tr_ids) // 5]); fit_ids = set(tr_ids[len(tr_ids) // 5:])
    G = {k: g for k, g in d.groupby("track_id")}
    print(f"tracks: fit {len(fit_ids)}, val {len(val_ids)}, test {len(te_ids)}", flush=True)

    def seqs(ids, level, seed):
        r = np.random.default_rng(seed); out = []
        for i in sorted(ids):
            g = degrade(G[i], level, r)
            if len(g) < 3: continue
            F, a, dt = features(g, ap); out.append((F, a, dt, g.state.values))
        return out

    hmm = ActivityHMM().fit(seqs(fit_ids, 0, 1))
    val = {lvl: seqs(val_ids, lvl, 2) for lvl in LEVELS}
    hmm.beta = hmm.fit_beta({l: v for l, v in val.items() if l > 0}); hmm.beta[0] = 1.0
    print("beta (clean-trained HMM)", hmm.beta, flush=True)
    # level-matched HMM: emissions refit on training tracks degraded to each level (level is known online from NACp/gaps)
    matched = {lvl: ActivityHMM().fit(seqs(fit_ids, lvl, 20 + lvl)) for lvl in LEVELS}
    for lvl, m in matched.items():
        m.Q, m.pi = hmm.Q, hmm.pi  # dynamics come from clean labels
        m.beta = m.fit_beta({lvl: val[lvl]})
    print("beta (matched)", {l: m.beta[l] for l, m in matched.items()}, flush=True)

    # per-fix gradient boosting (same features, no temporal model), trained on all levels
    Xs, ys = [], []
    for lvl in LEVELS:
        for F, a, dt, l in seqs(fit_ids, lvl, 10 + lvl):
            Xs.append(np.column_stack([F, a, np.log1p(dt)])); ys.append(l)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(np.vstack(Xs), np.concatenate(ys))
    prior = (np.bincount(np.concatenate(ys), minlength=K) + 1) / (len(np.concatenate(ys)) + K)

    def gbm_proba(F, a, dt):
        p = gbm.predict_proba(np.column_stack([F, a, np.log1p(dt)]))
        out = np.full((len(F), K), 1e-6); out[:, gbm.classes_] = p
        return out / out.sum(1, keepdims=True)

    # hybrid: discriminative scaled likelihood p(s|x)/p(s) inside the continuous-time filter
    hyb_ll = lambda F, a, dt: np.log(gbm_proba(F, a, dt)) - np.log(prior)
    hyb_beta = {}
    for lvl in LEVELS:
        hyb_beta[lvl] = hmm.fit_beta({lvl: val[lvl]}, hyb_ll)[lvl]
    print("beta (hybrid)", hyb_beta, flush=True)

    res, rel = {}, {}
    for lvl in LEVELS:
        r_ = np.random.default_rng(100 + lvl); test, test_t = [], []
        for i in sorted(te_ids):
            g = degrade(G[i], lvl, r_)
            if len(g) < 3: continue
            F, a, dt = features(g, ap); test.append((F, a, dt, g.state.values)); test_t.append(g.t.values)
        y = np.concatenate([s[3] for s in test])
        Ps = {
            "rules": np.vstack([rule_classifier(F, a) for F, a, _, _ in test]),
            "gbm_per_fix": np.vstack([gbm_proba(F, a, dt) for F, a, dt, _ in test]),
            "hmm_clean": np.vstack([hmm.filter(F, a, dt, 1.0) for F, a, dt, _ in test]),
            "hmm_clean_tempered": np.vstack([hmm.filter(F, a, dt, hmm.beta.get(lvl, 1.0)) for F, a, dt, _ in test]),
            "hmm_matched": np.vstack([matched[lvl].filter(F, a, dt, matched[lvl].beta[lvl]) for F, a, dt, _ in test]),
            "hybrid_gbm_hmm": np.vstack([hmm.filter(F, a, dt, hyb_beta[lvl], hyb_ll(F, a, dt)) for F, a, dt, _ in test]),
        }
        res[lvl] = {m: metrics(P, y) for m, P in Ps.items()}
        rel[lvl] = {m: reliability(P, y) for m, P in Ps.items() if m != "rules"}
        # takeoff-roll detection delay: first time P(roll) > 0.5 after the true roll start
        res[lvl]["roll_detect_delay_s"] = {}
        cuts = np.cumsum([len(s[3]) for s in test])[:-1]
        for m in ["rules", "gbm_per_fix", "hybrid_gbm_hmm"]:
            delays, missed = [], 0
            for (F, a, dt, l), P, ts in zip(test, np.split(Ps[m], cuts), test_t):
                on = np.where(l == S["TAKEOFF_ROLL"])[0]
                if not len(on): continue
                hit = np.where((P[:, S["TAKEOFF_ROLL"]] > 0.5) & (np.arange(len(l)) >= on[0]))[0]
                if len(hit): delays.append(ts[hit[0]] - ts[on[0]])
                else: missed += 1
            res[lvl]["roll_detect_delay_s"][m] = {"median": float(np.median(delays)) if delays else None,
                                                  "p90": float(np.percentile(delays, 90)) if delays else None,
                                                  "detected": len(delays), "missed": missed}
        res[lvl]["n_fixes"] = int(len(y))
        print(f"level {lvl}: " + "  ".join(f"{m}: acc {r['acc']:.3f} F1 {r['macro_f1']:.3f} LL {r['log_loss']:.3f} ECE {r['ece']:.3f}"
                                          for m, r in res[lvl].items() if isinstance(r, dict) and "acc" in r), flush=True)
        print("   roll detect delay", res[lvl]["roll_detect_delay_s"], flush=True)
    json.dump({"metrics": res, "reliability": rel, "beta": {"clean": hmm.beta, "hybrid": hyb_beta,
               "matched": {l: m.beta[l] for l, m in matched.items()}},
               "Q_mean_dwell_s": {STATES[k]: float(-1 / hmm.Q[k, k]) for k in range(K)}},
              open(ROOT / "reports/activity.json", "w"), indent=1)


if __name__ == "__main__":
    main()

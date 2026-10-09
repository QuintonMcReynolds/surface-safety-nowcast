"""Figures for the README from reports/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "reports", ROOT / "reports/figures"
F.mkdir(parents=True, exist_ok=True)
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "legend.frameon": False, "lines.linewidth": 2})
LEVEL_NAMES = ["as received", "½ rate, 10 m", "¼ rate, 25 m,\n30 s outages", "15 % rate, 50 m,\nno speed"]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def label_end(ax, xs, ys, text, color):
    ax.annotate(text, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.plot(xs[-1], ys[-1], "o", color=color, ms=4)


def activity():
    a = load("activity")
    if not a: return
    m = a["metrics"]; lv = sorted(m, key=int)
    models = [("rules", "Threshold rules"), ("gbm_per_fix", "GBM per fix"),
              ("hmm_clean_tempered", "CT-HMM, tempered"), ("hybrid_gbm_hmm", "Hybrid GBM + CT-HMM")]
    models = [x for x in models if x[0] in m[lv[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (key, title) in zip(axes, [("log_loss", "Log loss (lower is better)"), ("ece", "Calibration error, ECE (lower is better)")]):
        for i, (k, name) in enumerate(models):
            ys = [m[l][k][key] for l in lv]
            ax.plot(range(len(lv)), ys, color=SER[i], marker="o", ms=4, label=name)
        ax.set_xticks(range(len(lv)), LEVEL_NAMES[:len(lv)], fontsize=8); ax.set_title(title, loc="left", fontsize=10)
        if key == "log_loss": ax.set_yscale("log")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Activity inference as surveillance degrades (Oct 1 test day)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(F / "activity_degradation.png", dpi=160); plt.close(fig)
    rel = a.get("reliability", {})
    if rel:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for ax, l in zip(axes, ["0", "2"]):
            ax.plot([0, 1], [0, 1], color=GRID, lw=1.5)
            for i, (k, name) in enumerate(models):
                if k not in rel.get(l, {}): continue
                pts = rel[l][k]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SER[i], marker="o", ms=4, label=name)
            ax.set_title(f"Level {l}: {LEVEL_NAMES[int(l)].replace(chr(10), ' ')}", loc="left", fontsize=10)
            ax.set_xlabel("Stated confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        axes[0].set_ylabel("Observed accuracy"); axes[0].legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(F / "activity_reliability.png", dpi=160); plt.close(fig)


def eta():
    e = load("eta")
    if not e: return
    lvl = e["0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    names = [("physics", "Physics baseline"), ("quantile_gbm", "Quantile GBM"), ("cqr_static", "Conformal (static)"), ("cqr_aci", "Adaptive conformal")]
    for ax, tgt, title in zip(axes, ["y_enter", "y_roll"], ["Time to runway entry", "Time to takeoff roll"]):
        for i, (k, name) in enumerate(names):
            c = lvl[tgt][k]["coverage80_by_test_hour"]
            hrs = sorted(c, key=int)
            ax.plot([int(h) for h in hrs], [c[h] for h in hrs], color=SER[i], marker="o", ms=4, label=name)
        ax.axhline(0.8, color=INK2, lw=1, ls="--"); ax.text(0, 0.81, "80 % target", color=INK2, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("Test-day hour (12:00–18:00 UTC)"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Share of outcomes inside the 80 % interval"); axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(F / "eta_coverage.png", dpi=160); plt.close(fig)


def incursion():
    inc = load("incursion")
    if not inc: return
    kinds = [("geofence", "Geofence rule"), ("kinematic", "Required-deceleration rule"), ("gbm", "Learned (GBM)"), ("clearance_norm", "Clearance + braking norm")]
    lvls = [l for l in ["0", "2"] if l in inc]
    fig, axes = plt.subplots(1, len(lvls), figsize=(5 * len(lvls), 3.8), sharey=True, squeeze=False)
    for ax, l in zip(axes[0], lvls):
        for i, (k, name) in enumerate(kinds):
            if k not in inc[l]: continue
            cur = sorted(inc[l][k]["curve"], key=lambda c: c["false_alert_rate"])
            ax.plot([c["false_alert_rate"] for c in cur], [c["lead_median_s"] for c in cur], color=SER[i], marker="o", ms=3, label=name, drawstyle="steps-post")
        ax.set_title(f"Level {l}: {LEVEL_NAMES[int(l)].replace(chr(10), ' ')}", loc="left", fontsize=10)
        ax.set_xlabel("Share of real, correct stops that raised an alert"); ax.set_xlim(-0.02, 1.0)
    axes[0][0].set_ylabel("Median alert lead time (s)"); axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(F / "incursion_tradeoff.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    activity(); eta(); incursion()
    print(sorted(p.name for p in F.glob("*.png")))

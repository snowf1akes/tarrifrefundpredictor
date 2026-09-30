"""Generate all figures in figs/ by driving the compiled pricer. Usage: python plot.py [paths]"""
import math, os, re, subprocess, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BIN = next((b for b in ("./tariff-pricer.exe", "./tariff-pricer") if os.path.exists(b)), None)
assert BIN, "build first: g++ -std=c++17 -O3 -march=native -fopenmp main.cpp -o tariff-pricer"
PATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
FACE, CARRY, OFFER = 1_000_000, 25_000, 0.40
os.makedirs("figs", exist_ok=True)

# palette: categorical in fixed order, one-hue sequential ramp, text tokens for all labels
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "font.size": 10, "lines.linewidth": 2})

def run(csv=None, **kw):
    """Run the pricer with --key value flags; return parsed stats (and samples if csv)."""
    args = [BIN, "--paths", str(PATHS), "--carry", str(CARRY)] + [s for k, v in kw.items() for s in (f"--{k}", str(v))]
    if csv: args += ["--csv", csv]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    g = lambda k: float(re.search(rf"{re.escape(k)}\s+([-\d.]+)", out).group(1))
    d = {"mean": g("E[PV]"), "std": g("stddev"), "disc": g("E[disc]"), "carry": g("E[carry]")}
    if csv: d["pv"], d["t"] = np.loadtxt(csv, delimiter=",", skiprows=1, unpack=True)
    return d

def save(name):
    plt.tight_layout(); plt.savefig(f"figs/{name}.png", dpi=130); plt.close(); print("wrote figs/" + name + ".png")

tmp = os.path.join(tempfile.gettempdir(), "tariff_samples.csv")
base = run(csv=tmp)
pv, t = base["pv"] / FACE, base["t"]

# 1. PV distribution: full mass incl. losses (carry paid on lost claims)
plt.figure(figsize=(8, 4.2))
plt.hist(pv, bins=120, color=C[0], linewidth=0)
for x, lab, ls in [(pv.mean(), f"E[PV] = {pv.mean():.3f}", "-"), (np.percentile(pv, 5), "p5", ":"), (np.percentile(pv, 95), "p95", ":")]:
    plt.axvline(x, color=INK, ls=ls, lw=1.2, label=lab)
plt.axvline(OFFER, color=C[1], lw=1.5, label=f"cash offer = {OFFER:.2f}")
plt.title(f"Discounted PV per path ({PATHS:,} paths)   P(PV<0) = {np.mean(pv < 0):.2f}", loc="left")
plt.xlabel("PV / face"); plt.ylabel("paths"); plt.legend()
save("pv_distribution")

# 2. Monte Carlo convergence: running mean with 95% CI
n = np.arange(1, len(pv) + 1); rm = np.cumsum(pv) / n; ci = 1.96 * pv.std() / np.sqrt(n)
plt.figure(figsize=(8, 4.2))
plt.fill_between(n, rm - ci, rm + ci, color=C[0], alpha=0.18, linewidth=0, label="95% CI")
plt.plot(n, rm, color=C[0], label="running mean")
plt.axhline(pv.mean(), color=INK, lw=1, ls="--", label=f"final = {pv.mean():.4f}")
plt.xscale("log"); plt.xlim(100, len(pv)); plt.ylim(pv.mean() - 0.05, pv.mean() + 0.05)
plt.title("Monte Carlo convergence of E[PV] / face", loc="left"); plt.xlabel("paths"); plt.ylabel("E[PV] / face"); plt.legend()
save("convergence")

# 3. Time-to-resolution models
plt.figure(figsize=(8, 4.2))
models = [("Exponential  (mean 2y)", {}), ("Weibull k=1.5  (peaked)", {"dist": "weibull", "shape": 1.5, "scale": 2.0}),
          ("Weibull k=0.7  (fat tail)", {"dist": "weibull", "shape": 0.7, "scale": 2.0})]
for i, (lab, kw) in enumerate(models):
    tt = run(csv=tmp, **kw)["t"]
    plt.hist(tt[tt < 10], bins=100, range=(0, 10), density=True, histtype="step", color=C[i], lw=2, label=lab)
plt.title("Stochastic time-to-resolution (inverse-CDF sampled)", loc="left")
plt.xlabel("years until resolution"); plt.ylabel("density"); plt.legend()
save("resolution_time")

# 4. Sell-vs-hold decision boundary: breakeven offer over P(win) x risk penalty
pw, rk = np.linspace(0.1, 0.9, 9), np.linspace(0, 2, 9)
grid = np.array([[(s := run(p=p))["mean"] - r * s["std"] for p in pw] for r in rk]) / FACE
plt.figure(figsize=(8, 4.6))
im = plt.pcolormesh(pw, rk, grid, cmap=SEQ, shading="nearest", edgecolors="#fcfcfb", linewidth=1)
plt.colorbar(im, label="certainty equivalent / face  (breakeven offer)")
cs = plt.contour(pw, rk, grid, levels=[OFFER], colors=[C[1]], linewidths=2)
plt.clabel(cs, fmt={OFFER: f"offer {OFFER:.2f}"}, colors=[INK])
plt.text(0.12, 1.85, "SELL", color=INK, weight="bold"); plt.text(0.8, 0.1, "HOLD", color=INK, weight="bold")
plt.grid(False); plt.title("Sell-vs-hold boundary:  hold iff  mean - risk*stddev  >  offer", loc="left")
plt.xlabel("P(win)"); plt.ylabel("risk penalty (per unit stddev)")
save("decision_boundary")

# 5. Implied P(win) from observed market price, per discount rate
mkt = np.linspace(0.05, 0.95, 50)
plt.figure(figsize=(8, 4.2))
for i, rate in enumerate([0.02, 0.05, 0.10]):
    s = run(rate=rate)
    plt.plot(mkt, np.clip((mkt * FACE + s["carry"]) / (FACE * s["disc"]), 0, 1), color=C[i], label=f"rate = {rate:.0%}")
plt.axhline(1, color=INK2, lw=0.8, ls=":")
plt.title("Implied P(win)  =  (price + E[carry]) / (face * E[exp(-rT)])", loc="left")
plt.xlabel("market price / face"); plt.ylabel("implied P(win)"); plt.legend()
save("implied_probability")

# 6. Sensitivity: E[PV] vs expected resolution time, by tail shape (Weibull scale set so mean matches)
means = np.linspace(0.5, 6, 12)
plt.figure(figsize=(8, 4.2))
for i, (lab, k) in enumerate([("Exponential", None), ("Weibull k=1.5", 1.5), ("Weibull k=0.7", 0.7)]):
    kw = {} if k is None else {"dist": "weibull", "shape": k}
    ys = [run(scale=m if k is None else m / math.gamma(1 + 1 / k), **kw)["mean"] / FACE for m in means]
    plt.plot(means, ys, color=C[i], marker="o", ms=4, label=lab)
plt.axhline(OFFER, color=INK, lw=1, ls="--", label=f"cash offer {OFFER:.2f}")
plt.title(f"E[PV] vs expected time-to-resolution  (carry {CARRY:,}/yr, rate 5%)", loc="left")
plt.xlabel("E[T] years"); plt.ylabel("E[PV] / face"); plt.legend()
save("sensitivity")

"""Histogram of PV samples from `tariff-pricer --csv samples.csv`. Usage: python plot.py samples.csv [out.png]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pv = np.loadtxt(sys.argv[1], skiprows=1)
out = sys.argv[2] if len(sys.argv) > 2 else "pv_hist.png"
plt.hist(pv[pv > 0], bins=100, color="steelblue")
plt.axvline(pv.mean(), color="red", label=f"E[PV]={pv.mean():,.0f}  (P(0)={np.mean(pv == 0):.2f})")
plt.xlabel("discounted PV"); plt.ylabel("paths"); plt.legend(); plt.title("Tariff-refund claim PV (win paths)")
plt.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out)

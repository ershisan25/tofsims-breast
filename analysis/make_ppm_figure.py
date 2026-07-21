"""Supplementary QC figure: mass-calibration accuracy AFTER preprocessing.

For every calibrated/denoised/baseline-corrected/normalized spectrum we take the
detected peak nearest each reference calibrant (within the analysed m/z 50-250
window) and compute the ppm error (observed - theoretical)/theoretical x1e6.
Data source: peak_analysis/all4_50-250/all_detected_peaks.csv (this project)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
PEAKS = HERE.parent / "peak_analysis" / "all4_50-250" / "all_detected_peaks.csv"
FIG = HERE / "figures"

# reference calibrants (theoretical m/z) that fall inside the analysed window
REF = {57.0698767102: "C4H9", 58.065674: "C3H8N", 67.0548: "C5H7", 71.0855267746: "C5H11",
       86.096976: "C5H12N", 91.0542266457: "C7H7", 104.107539: "C5H14NO",
       184.073320: "C5H15NO4P", 224.105171: "C8H19NO4P"}
TOL_PPM = 300   # accept the closest detected peak only if within +-300 ppm (else it is a different ion)
TCOL = {"Breast": "#e74c3c", "Serum": "#9467bd", "Liver": "#3498db", "Fur": "#2ca02c"}

d = pd.read_csv(PEAKS)
d["tissue"] = d.SampleName.str.split("_").str[0].map(
    {"breast": "Breast", "blood serum": "Serum", "liver": "Liver", "Fur": "Fur"})
d["sample"] = d.SampleName.str.replace("_calibrated_denoised_baseline_corrected_normalized", "", regex=False)

rows = []
refs = np.array(sorted(REF))
for (samp, tis), g in d.groupby(["sample", "tissue"]):
    pc = g.PeakCenter.to_numpy()
    for ref in refs:
        if pc.size == 0:
            continue
        obs = pc[np.argmin(np.abs(pc - ref))]
        ppm = (obs - ref) / ref * 1e6
        if abs(ppm) <= TOL_PPM:
            rows.append({"tissue": tis, "ref": ref, "ppm": ppm})
p = pd.DataFrame(rows)
print(f"{len(p)} calibrant observations across {p.tissue.nunique()} tissues")
print("median |ppm| per tissue:\n", p.assign(abs_ppm=p.ppm.abs()).groupby("tissue").abs_ppm.median().round(1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), width_ratios=[1.6, 1.0], constrained_layout=True)

# A — signed ppm error per reference calibrant (box = all samples), jittered points by tissue
labels = [f"{r:.2f}\n{REF[r]}$^+$" for r in refs]
box = ax1.boxplot([p[p.ref.eq(r)].ppm.values for r in refs], positions=range(len(refs)),
                  widths=0.6, showfliers=False, patch_artist=True)
for b in box["boxes"]:
    b.set_facecolor("#eeeeee"); b.set_edgecolor("#555555")
for k in ("medians",):
    for m in box[k]: m.set_color("#000000")
rng = np.random.default_rng(0)
for i, r in enumerate(refs):
    sub = p[p.ref.eq(r)]
    ax1.scatter(i + (rng.random(len(sub)) - 0.5) * 0.4, sub.ppm, s=6,
                c=[TCOL[t] for t in sub.tissue], alpha=0.45, linewidths=0)
ax1.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6)
ax1.set(xticks=range(len(refs)), ylabel="Mass error (ppm)",
        xlabel="Reference calibrant (m/z, assignment)")
ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_title("A  Mass accuracy after preprocessing, per reference calibrant", loc="left", fontweight="bold", fontsize=12, pad=6)
ax1.legend(handles=[plt.Line2D([0], [0], marker="o", ls="none", mfc=c, mec="none", label=t)
                    for t, c in TCOL.items()], frameon=False, fontsize=8, loc="upper right", ncol=2)

# B — median absolute ppm per tissue
med = p.assign(a=p.ppm.abs()).groupby("tissue").a.median().reindex(["Breast", "Serum", "Liver", "Fur"])
ax2.bar(range(len(med)), med.values, color=[TCOL[t] for t in med.index], alpha=0.85)
for i, v in enumerate(med.values):
    ax2.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
ax2.set(xticks=range(len(med)), xticklabels=med.index, ylabel="Median |mass error| (ppm)")
ax2.set_title("B  Median absolute error by tissue", loc="left", fontweight="bold", fontsize=12, pad=6)

fig.savefig(FIG / "fig_S_ppm_accuracy.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_S_ppm_accuracy.png", bbox_inches="tight")
print("wrote fig_S_ppm_accuracy")

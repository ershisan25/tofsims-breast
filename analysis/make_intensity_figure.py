"""SI figure justifying the m/z 50-250 analysis window: within the aligned peak
table, low-mass ions are more intense and more reproducibly detected than
higher-mass ions.  Data: peak_analysis/all4_50-250/peak_intensity_table.csv."""
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
FIG = HERE / "figures"
d = pd.read_csv(HERE.parent / "peak_analysis" / "all4_50-250" / "peak_intensity_table.csv")
mz = [c for c in d.columns if c not in ("SampleName", "Group")]
mzf = np.array([float(m) for m in mz])
X = d[mz].to_numpy(float)
mean_int = X.mean(0)
cv = np.where(mean_int > 0, X.std(0) / mean_int, np.nan)
detf = (X > 0).mean(0)

# binned medians (50-Da bands) for the trend lines
bands = [(50, 100), (100, 150), (150, 200), (200, 250)]
bctr = [(lo + hi) / 2 for lo, hi in bands]

def band_med(v):
    return [np.nanmedian(v[(mzf >= lo) & (mzf < hi)]) for lo, hi in bands]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
BLUE = "#1f77b4"

# A — peak intensity vs m/z
ax1.scatter(mzf, np.log10(mean_int + 1), s=16, c=BLUE, alpha=0.45, edgecolors="none")
ax1.plot(bctr, band_med(np.log10(mean_int + 1)), "-o", color="#d62728", lw=2, label="50-Da band median")
ax1.set(xlabel="m/z", ylabel="log10(mean peak intensity + 1)")
ax1.set_title("A  Peak intensity decreases with m/z", loc="left", fontweight="bold", fontsize=12, pad=6)
ax1.legend(frameon=False, fontsize=9)

# B — detection reproducibility vs m/z
ax2.scatter(mzf, detf, s=16, c=BLUE, alpha=0.45, edgecolors="none")
ax2.plot(bctr, band_med(detf), "-o", color="#d62728", lw=2, label="50-Da band median")
ax2.set(xlabel="m/z", ylabel="Detection frequency across spectra", ylim=(-0.03, 1.03))
ax2.set_title("B  Detection reproducibility decreases with m/z", loc="left", fontweight="bold", fontsize=12, pad=6)
ax2.legend(frameon=False, fontsize=9)

fig.savefig(FIG / "fig_S_intensity_mass.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_S_intensity_mass.png", bbox_inches="tight")
print("wrote fig_S_intensity_mass |",
      "band intensity:", [round(v, 2) for v in band_med(np.log10(mean_int + 1))],
      "| band detection:", [round(v, 2) for v in band_med(detf)])

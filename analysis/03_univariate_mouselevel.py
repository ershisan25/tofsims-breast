"""Mouse-level univariate differential-ion analysis (fixes A2/A7).

For each tissue: aggregate spots to ONE value per animal (per-feature mean),
then Welch t-test + Mann-Whitney across ANIMALS, BH-FDR.  Contrast the number
of 'significant differential ions' against the naive spot-level test the paper
used.  Also flags features whose signal is driven by DETECTION RATE rather than
intensity (A7).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import lib_analysis as L

df, mass = L.load()
TISSUES = ['Breast tissue', 'Fur', 'Blood serum', 'Liver']
ALPHA = 0.05

summary = []
for tis in TISSUES:
    sub = df[df['Tissue'] == tis]
    Xr = sub[mass]
    grp = sub['Group'].values

    # ---- naive SPOT-level (their approach) ----
    def count_sig(Xmat, labels):
        pv = []
        cm, ctm = labels == 'Cancer', labels == 'Control'
        for f in range(Xmat.shape[1]):
            a, b = Xmat[cm, f], Xmat[ctm, f]
            if len(a) < 3 or len(b) < 3 or (a.std() == 0 and b.std() == 0):
                pv.append(1.0); continue
            pv.append(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        pv = np.nan_to_num(np.array(pv), nan=1.0)
        fdr = multipletests(pv, method='fdr_bh')[1]
        return pv, fdr

    p_spot, fdr_spot = count_sig(Xr.values, grp)

    # ---- mouse-level: aggregate to per-animal mean ----
    agg = sub.groupby('AnimalID').agg(
        Group=('Group', 'first'), **{f: (f, 'mean') for f in mass}).reset_index()
    Xa = agg[mass].values
    ga = agg['Group'].values
    p_mouse, fdr_mouse = count_sig(Xa, ga)

    # detection-rate difference per feature (A7 flag)
    det_c = (Xr.values[grp == 'Cancer'] > 0).mean(0)
    det_t = (Xr.values[grp == 'Control'] > 0).mean(0)
    det_diff = np.abs(det_c - det_t)

    out = pd.DataFrame({'Feature': mass, 'p_spot': p_spot, 'FDR_spot': fdr_spot,
                        'p_mouse': p_mouse, 'FDR_mouse': fdr_mouse,
                        'detect_diff': det_diff})
    out.sort_values('p_mouse').round(5).to_csv(f'results/univariate_{tis.split()[0].lower()}.csv', index=False)

    n_spot = int((fdr_spot < ALPHA).sum())
    n_mouse = int((fdr_mouse < ALPHA).sum())
    # of the mouse-significant, how many are mostly detection-rate driven (>30% gap)
    n_detdriven = int(((fdr_mouse < ALPHA) & (det_diff > 0.30)).sum())
    n_c = (ga == 'Cancer').sum(); n_ct = (ga == 'Control').sum()
    summary.append(dict(Tissue=tis, animals=f'{n_c}v{n_ct}', spots=len(grp),
                        sig_ions_SPOT=n_spot, sig_ions_MOUSE=n_mouse,
                        of_which_detection_driven=n_detdriven))

s = pd.DataFrame(summary)
print("\n===== DIFFERENTIAL IONS: spot-level (paper) vs mouse-level (FDR<0.05) =====")
print(s.to_string(index=False))
s.to_csv('results/univariate_summary.csv', index=False)
print("\nPer-tissue full tables: results/univariate_<tissue>.csv")
print("sig_ions_SPOT = anticonservative count the paper reports; sig_ions_MOUSE = honest count.")

"""Publication figures for the analysis (fixes figure section C).

Okabe-Ito colorblind-safe palette; vector PDF + 300dpi PNG; sample sizes in
captions/labels; mouse-level everything.
Produces:
  fig_A_cross_tissue_auc.pdf   forest plot: mouse-level AUC + 95% CI + perm p
  fig_B_volcano_breast.pdf     fixed volcano (mouse-level, -log10 FDR)
  fig_C_roc_ci.pdf             ROC w/ bootstrap CI band, breast vs fur
  fig_D_pseudoreplication.pdf  sig-ion count spot vs mouse
  fig_E_stability_breast.pdf   fold-wise selection frequency (stable signature)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
import lib_analysis as L

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'pdf.fonttype': 42, 'ps.fonttype': 42})
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
      'purple': '#CC79A7', 'grey': '#999999', 'black': '#000000'}

def save(fig, name):
    fig.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'figures/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

df, mass = L.load()
auc = pd.read_csv('results/classification_auc.csv')
perm = pd.read_csv('results/permutation_test.csv')
uni = pd.read_csv('results/univariate_summary.csv')
stab = pd.read_csv('results/stability_selection_breast.csv')

# ---------- Fig A: cross-tissue AUC forest (RF) ----------
sub = auc[auc.Model == 'RandomForest'].set_index('Tissue').loc[
    ['Breast tissue', 'Fur', 'Liver', 'Blood serum']]
permm = perm.set_index('Tissue')
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ypos = np.arange(len(sub))[::-1]
for yp, (tis, r) in zip(ypos, sub.iterrows()):
    p = permm.loc[tis, 'p_perm']
    col = OI['green'] if p < 0.05 else OI['grey']
    ax.plot([r.CI_lo, r.CI_hi], [yp, yp], color=col, lw=2.5, solid_capstyle='round')
    ax.plot(r.AUC_mouse, yp, 'o', color=col, ms=8, zorder=3)
    lab = f"{tis}\n({int(r.n_cancer_an)}v{int(r.n_ctrl_an)} mice, {int(r.n_spots)} spots)"
    ax.text(-0.02, yp, lab, ha='right', va='center', fontsize=8.5)
    star = '  p={:.3f}{}'.format(p, '  *' if p < 0.05 else '  n.s.')
    ax.text(r.CI_hi + 0.02, yp, f"AUC={r.AUC_mouse:.2f} [{r.CI_lo:.2f}, {r.CI_hi:.2f}]{star}",
            va='center', fontsize=8)
ax.axvline(0.5, color=OI['red'], ls='--', lw=1.2, label='chance (0.5)')
ax.set_xlim(0, 1.0); ax.set_yticks([]); ax.set_xlabel('Mouse-level AUC (leave-one-animal-out) ± 95% CI')
ax.set_title('Cancer/control discrimination by tissue (Random Forest)\nonly breast is above chance',
             fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, frameon=False)
save(fig, 'fig_A_cross_tissue_auc')

# ---------- Fig B: fixed volcano, breast, mouse-level ----------
bu = pd.read_csv('results/univariate_breast.csv')
# mouse-level fold change from animal means
agg = df[df.Tissue == 'Breast tissue'].groupby('AnimalID').agg(
    Group=('Group', 'first'), **{f: (f, 'mean') for f in mass}).reset_index()
c = agg[agg.Group == 'Cancer'][mass].mean(); t = agg[agg.Group == 'Control'][mass].mean()
l2fc = np.log2((c + 1e-6) / (t + 1e-6))
vol = bu.set_index('Feature').join(l2fc.rename('l2fc'))
vol['neglogFDR'] = -np.log10(vol['FDR_mouse'].clip(lower=1e-300))
sig = vol['FDR_mouse'] < 0.05
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(vol.loc[~sig, 'l2fc'], vol.loc[~sig, 'neglogFDR'], s=14, c=OI['grey'], alpha=.5, label='n.s.')
up = sig & (vol.l2fc > 0); dn = sig & (vol.l2fc < 0)
ax.scatter(vol.loc[up, 'l2fc'], vol.loc[up, 'neglogFDR'], s=18, c=OI['red'], label='up in cancer (FDR<0.05)')
ax.scatter(vol.loc[dn, 'l2fc'], vol.loc[dn, 'neglogFDR'], s=18, c=OI['blue'], label='down in cancer (FDR<0.05)')
ax.axhline(-np.log10(0.05), color='k', ls='--', lw=1)
ax.set_xlabel('log2 fold change (cancer/control), mouse means')
ax.set_ylabel('−log10(FDR)   [BH-corrected]')
ax.set_title('Breast: differential ions (mouse-level, n=19v19)\naxis & threshold both on FDR', fontsize=9.5, fontweight='bold')
ax.legend(fontsize=7.5, frameon=False)
save(fig, 'fig_B_volcano_breast')

# ---------- Fig C: ROC with bootstrap CI band, breast vs fur (RF) ----------
def animal_roc(tis):
    X, y, groups, _ = L.tissue_slice(df, mass, tis)
    pipe = Pipeline([('prep', L.SpectraPreproc(0.2)), ('clf', L.rf())])
    a, _ = L.animal_level_scores(X, y, groups, pipe)
    yv, pv = a['y'].values, a['prob'].values
    base = np.linspace(0, 1, 100); tprs = []
    rng = np.random.default_rng(42)
    ip, ineg = np.where(yv == 1)[0], np.where(yv == 0)[0]
    for _ in range(1000):
        bi = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
        if len(np.unique(yv[bi])) < 2: continue
        fpr, tpr, _ = roc_curve(yv[bi], pv[bi]); tprs.append(np.interp(base, fpr, tpr))
    tprs = np.array(tprs); fpr, tpr, _ = roc_curve(yv, pv)
    return base, np.interp(base, fpr, tpr), np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), roc_auc_score(yv, pv)
fig, ax = plt.subplots(figsize=(4.4, 4.2))
for tis, col in [('Breast tissue', OI['green']), ('Fur', OI['orange'])]:
    b, m, lo, hi, a = animal_roc(tis)
    ax.plot(b, m, color=col, lw=2, label=f'{tis} (AUC={a:.2f})')
    ax.fill_between(b, lo, hi, color=col, alpha=.18)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
ax.set_title('Mouse-level ROC ± 95% bootstrap band', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, frameon=False)
save(fig, 'fig_C_roc_ci')

# ---------- Fig D: pseudoreplication effect on sig-ion counts ----------
fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(uni)); w = 0.38
ax.bar(x - w/2, uni['sig_ions_SPOT'], w, color=OI['grey'], label='spot-level (paper)')
ax.bar(x + w/2, uni['sig_ions_MOUSE'], w, color=OI['blue'], label='mouse-level (correct)')
for i, r in uni.iterrows():
    ax.text(i - w/2, r.sig_ions_SPOT + 1, int(r.sig_ions_SPOT), ha='center', fontsize=8)
    ax.text(i + w/2, r.sig_ions_MOUSE + 1, int(r.sig_ions_MOUSE), ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([t.replace(' tissue', '').replace('Blood ', '') for t in uni['Tissue']])
ax.set_ylabel('# differential ions (FDR<0.05)')
ax.set_title('Pseudoreplication inflates significance\n(peripheral tissues: all artifacts)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, frameon=False)
save(fig, 'fig_D_pseudoreplication')

# ---------- Fig E: stability selection, breast ----------
top = stab.sort_values('mean_freq', ascending=False).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(4.8, 4.2))
yp = np.arange(len(top))
ax.barh(yp - 0.2, top['RF_selfreq'], 0.4, color=OI['green'], label='Random Forest')
ax.barh(yp + 0.2, top['Univ_selfreq'], 0.4, color=OI['purple'], label='Univariate')
ax.axvline(0.9, color=OI['red'], ls='--', lw=1, label='90% stability')
ax.set_yticks(yp); ax.set_yticklabels([f'm/z {f}' for f in top['Feature']], fontsize=8)
ax.set_xlabel('Fold selection frequency (38 folds)')
ax.set_title('Breast signature stability (fold-wise)\nstable set: m/z 52,53,54,66,67,80,118', fontsize=9.5, fontweight='bold')
ax.legend(fontsize=8, frameon=False, loc='lower right')
save(fig, 'fig_E_stability_breast')

print("Figures written to figures/*.pdf and *.png:")
import os
for f in sorted(os.listdir('figures')):
    if f.endswith('.pdf'): print("  ", f)

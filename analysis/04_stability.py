"""Fold-wise stability selection of the breast signature (fixes A4).

Within each leave-one-animal-out fold, refit RF and univariate on TRAIN ONLY,
take each method's top-15 ions, and count how often each ion is selected across
the 38 folds.  Ions selected in >=90% of folds = the defensible 'stable
signature'.  Compare to the paper's claimed signature (m/z 54, 67, 69, 157) and
to in-sample selection.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import ttest_ind
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
import lib_analysis as L


def pls_vip(Xk, ytr, ncomp=5):
    pls = PLSRegression(n_components=ncomp, scale=False).fit(Xk, ytr.astype(float))
    ts, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = (q[0] ** 2) * (ts ** 2).sum(axis=0)
    return np.sqrt(w.shape[0] * ((w / np.linalg.norm(w, axis=0)) ** 2 @ ss) / ss.sum())

df, mass = L.load()
X, y, groups, sub = L.tissue_slice(df, mass, 'Breast tissue')
mass_arr = np.array(mass)
TOPK = 15
logo = LeaveOneGroupOut()
# RF + univariate define the 'stable signature' (used in Fig S1 + main text);
# PLS-DA VIP is computed too, but only *displayed* in Fig S4 (Wei's choice).
rf_hits = np.zeros(len(mass)); uni_hits = np.zeros(len(mass)); pls_hits = np.zeros(len(mass)); nfold = 0

for tr, te in logo.split(X, y, groups):
    Xtr = X.iloc[tr].values; ytr = y[tr]
    prep = L.SpectraPreproc(0.2).fit(Xtr)
    keep = prep.keep_
    Xk = prep.transform(Xtr)
    kept_idx = np.where(keep)[0]
    rf = L.rf().fit(Xk, ytr)
    rf_hits[kept_idx[np.argsort(rf.feature_importances_)[::-1][:TOPK]]] += 1
    cm, ctm = ytr == 1, ytr == 0
    tp = np.array([ttest_ind(Xk[cm, j], Xk[ctm, j], equal_var=False).pvalue for j in range(Xk.shape[1])])
    uni_hits[kept_idx[np.argsort(np.nan_to_num(tp, nan=1))[:TOPK]]] += 1
    pls_hits[kept_idx[np.argsort(pls_vip(Xk, ytr))[::-1][:TOPK]]] += 1
    nfold += 1

stab = pd.DataFrame({'Feature': mass_arr,
                     'RF_selfreq': rf_hits / nfold,
                     'Univ_selfreq': uni_hits / nfold,
                     'PLS_selfreq': pls_hits / nfold})
stab['mean_freq'] = stab[['RF_selfreq', 'Univ_selfreq']].mean(1)
stab = stab.sort_values('mean_freq', ascending=False)
stab.round(3).to_csv('results/stability_selection_breast.csv', index=False)

stable = stab[(stab['RF_selfreq'] >= 0.9) & (stab['Univ_selfreq'] >= 0.9)]
print(f"\n===== BREAST STABILITY SELECTION ({nfold} folds, top-{TOPK}/method) =====")
print(f"Ions selected in >=90% of folds by BOTH RF and univariate: {len(stable)}")
print(stable[['Feature', 'RF_selfreq', 'Univ_selfreq', 'PLS_selfreq']].round(2).to_string(index=False))
print("\nTop 20 by mean selection frequency:")
print(stab.head(20)[['Feature', 'RF_selfreq', 'Univ_selfreq', 'PLS_selfreq', 'mean_freq']].round(2).to_string(index=False))
claimed = ['54.04', '67.05', '69.07', '157']
print(f"\nPaper's claimed signature ~ m/z 54,67,69,157 -> their fold selection freq:")
for c in ['54', '67', '69', '157']:
    m = stab[stab['Feature'].str.startswith(c)]
    if len(m):
        r = m.iloc[0]; print(f"  m/z {r['Feature']:8s} RF={r['RF_selfreq']:.2f} Univ={r['Univ_selfreq']:.2f} PLS={r['PLS_selfreq']:.2f}")
print("\nSaved: results/stability_selection_breast.csv")

"""Signature by cross-validation selection frequency at the ANIMAL level.
Within each leave-one-animal-out fold: aggregate TRAINING animals to one mean
profile each, fit preprocessing on those, then record top-15 ions by RF
importance and top-15 by Mann-Whitney rank test (both on animal-mean training
data). Report ions selected in >=90% of folds by BOTH criteria.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut
import lib_analysis as L


def pls_vip(Xk, ya, ncomp=5):
    pls = PLSRegression(n_components=ncomp, scale=False).fit(Xk, ya.astype(float))
    ts, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = (q[0] ** 2) * (ts ** 2).sum(axis=0)
    return np.sqrt(w.shape[0] * ((w / np.linalg.norm(w, axis=0)) ** 2 @ ss) / ss.sum())

df, mass = L.load()
X, y, groups, sub = L.tissue_slice(df, mass, 'Breast tissue')
mass_arr = np.array(mass); TOPK = 15
sub = sub.reset_index(drop=True)

logo = LeaveOneGroupOut()
rf_hits = np.zeros(len(mass)); uni_hits = np.zeros(len(mass)); pls_hits = np.zeros(len(mass)); nf = 0
for tr, te in logo.split(X, y, groups):
    train_df = sub.iloc[tr]
    agg = train_df.groupby('AnimalID').agg(
        yy=('Group', lambda s: int((s == 'Cancer').iloc[0])),
        **{f: (f, 'mean') for f in mass}).reset_index()
    Xa = agg[mass].values; ya = agg['yy'].values
    prep = L.SpectraPreproc(0.2).fit(Xa); ki = np.where(prep.keep_)[0]
    Xk = prep.transform(Xa)
    rf = L.rf().fit(Xk, ya)
    rf_hits[ki[np.argsort(rf.feature_importances_)[::-1][:TOPK]]] += 1
    pv = np.array([mannwhitneyu(Xk[ya == 1, j], Xk[ya == 0, j], alternative='two-sided').pvalue
                   for j in range(Xk.shape[1])])
    uni_hits[ki[np.argsort(np.nan_to_num(pv, nan=1))[:TOPK]]] += 1
    pls_hits[ki[np.argsort(pls_vip(Xk, ya))[::-1][:TOPK]]] += 1
    nf += 1

det = (X.values > 0).mean(0)
stab = pd.DataFrame({'Feature': mass_arr, 'RF_freq': rf_hits / nf,
                     'Univ_freq': uni_hits / nf, 'PLS_freq': pls_hits / nf, 'detect_rate': det})
stable = stab[(stab.RF_freq >= .9) & (stab.Univ_freq >= .9)].sort_values('RF_freq', ascending=False)
stab.sort_values('RF_freq', ascending=False).round(3).to_csv('results/stability_animal_breast.csv', index=False)
print(f"ANIMAL-LEVEL stability selection ({nf} folds, top-{TOPK}/criterion)")
print("Ions selected in >=90% of folds by BOTH RF and rank test:")
print(stable[['Feature', 'RF_freq', 'Univ_freq', 'PLS_freq', 'detect_rate']].round(2).to_string(index=False))
print("\nCompare to previous spot-level signature m/z 52,53,54,66,67,80,118:")
for c in ['52', '53', '54', '66', '67', '80', '118', '69', '157']:
    m = stab[stab.Feature.astype(str).str.match(rf'^{c}(\.|$)')]
    if len(m):
        r = m.iloc[0]; print(f"  m/z {r.Feature:8s} RF={r.RF_freq:.2f} Univ={r.Univ_freq:.2f} PLS={r.PLS_freq:.2f} det={r.detect_rate:.0%}")

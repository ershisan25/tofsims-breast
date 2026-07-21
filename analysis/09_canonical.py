"""Canonical fixes + the correct 'preservation' endpoint.
Fixes: (1) differential ions use Mann-Whitney (exact for small n), not Welch;
(2) detection assessed at the ANIMAL level (two-part); (3) implements the
signature-PRESERVATION test = lock the breast-trained model, apply it to the
peripheral tissues (transfer AUC + permutation), instead of separate per-tissue
classifiers. Cross-tissue AUC-difference bootstrap is NOT computed (invalid:
overlapping cohorts)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lib_analysis as L

df, mass = L.load()
TIS = ['Breast tissue', 'Fur', 'Liver', 'Blood serum']
rng = np.random.default_rng(L.RANDOM_STATE)

# ---------- (1) differential ions: Mann-Whitney, EXACT for small n ----------
def mwu_exact(a, b):
    try:
        m = 'exact' if (len(a) <= 12 and len(b) <= 12) else 'asymptotic'
        return mannwhitneyu(a, b, alternative='two-sided', method=m).pvalue
    except ValueError:
        return 1.0
rows = []
for t in TIS:
    s = df[df.Tissue == t]
    agg = s.groupby('AnimalID').agg(Group=('Group', 'first'), **{f: (f, 'mean') for f in mass}).reset_index()
    X = agg[mass].values; y = (agg.Group == 'Cancer').values
    pv = np.array([mwu_exact(X[y, j], X[~y, j]) for j in range(X.shape[1])])
    fdr = multipletests(np.nan_to_num(pv, nan=1.0), method='fdr_bh')[1]
    # animal-level detection two-part: fraction of animals with ion detected, per group
    detA = agg[mass].gt(0)
    dc = detA[y.tolist() if False else y].mean(0) if False else detA[agg.Group.values == 'Cancer'].mean(0)
    dt = detA[agg.Group.values == 'Control'].mean(0)
    det_diff = (dc.values - dt.values)
    rows.append(dict(Tissue=t, animals=f"{int(y.sum())}v{int((~y).sum())}",
                     sig_MWU=int((fdr < .05).sum()),
                     detection_driven=int(((fdr < .05) & (np.abs(det_diff) > .30)).sum())))
uni = pd.DataFrame(rows)
uni.to_csv('results/univariate_canonical.csv', index=False)
print("=== (1) Differential ions — Mann-Whitney (exact for small n), BH<0.05 ===")
print(uni.to_string(index=False))

# ---------- (3) PRESERVATION transfer test ----------
# Lock the breast-trained model, apply it to each peripheral tissue.
Xb, yb, gb, sb = L.tissue_slice(df, mass, 'Breast tissue')
def fit_breast(model):
    clf = L.PLSDAClassifier(5) if model == 'PLS-DA' else L.rf()
    pipe = Pipeline([('prep', L.SpectraPreproc(0.2)), ('clf', clf)]).fit(Xb, yb)
    return pipe
def animal_scores_apply(pipe, X, y, g):
    sc = pipe.predict_proba(X)[:, 1] if hasattr(pipe, 'predict_proba') else pipe.decision_function(X)
    a = pd.DataFrame({'g': g, 'y': y, 's': sc}).groupby('g').agg(y=('y', 'first'), s=('s', 'mean'))
    return a
def perm_transfer(pipe, X, y, g, obs, B=2000):
    a = pd.DataFrame({'g': g, 'y': y}).groupby('g')['y'].first()
    aids, al = a.index.values, a.values; stat = abs(obs - 0.5); hits = 0
    base = pipe.predict_proba(X)[:, 1]
    dd = pd.DataFrame({'g': g, 's': base})
    for _ in range(B):
        m = dict(zip(aids, rng.permutation(al)))
        aa = pd.DataFrame({'g': g, 'y': [m[x] for x in g], 's': base}).groupby('g').agg(y=('y', 'first'), s=('s', 'mean'))
        hits += abs(roc_auc_score(aa.y, aa.s) - 0.5) >= stat
    return (1 + hits) / (B + 1)

print("\n=== (3) PRESERVATION test — breast-locked model transferred to peripheral tissues ===")
prows = []
for model in ['PLS-DA', 'RandomForest']:
    pipe = fit_breast(model)
    for t in ['Fur', 'Liver', 'Blood serum']:
        X, y, g, s = L.tissue_slice(df, mass, t)
        a = animal_scores_apply(pipe, X.values if hasattr(X, 'values') else X, y, g)
        auc = roc_auc_score(a.y, a.s); p = perm_transfer(pipe, X.values, y, g, auc)
        prows.append(dict(Model=model, Tissue=t, transfer_AUC=round(auc, 3), perm_p=round(p, 4)))
        print(f"  breast-{model:12s} -> {t:12s}: transfer AUC={auc:.3f}  perm p={p:.4f}")
pres = pd.DataFrame(prows)
pres['perm_p_BH'] = multipletests(pres.perm_p, method='fdr_bh')[1].round(3)
pres.to_csv('results/preservation_transfer.csv', index=False)
print("\nInterpretation: transfer AUC>0.5 with sig p => the BREAST signature itself discriminates in that tissue")
print("(the correct test of 'preservation'). AUC~0.5 / n.s. => not preserved / undetectable.")
print("\nSaved: results/univariate_canonical.csv, results/preservation_transfer.csv")

"""Full per-model permutation test, SAME 500-tree RF for observed and null,
spot-trained/animal-scored, all 4 tissues x 2 models (incl. mammary-RF).
Two-sided |AUC-0.5|, N_PERM permutations, BH-adjusted across the 8-test family.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import lib_analysis as L

df, mass = L.load()
TISSUES = ['Breast tissue', 'Fur', 'Liver', 'Blood serum']
N_PERM = 500
rng = np.random.default_rng(L.RANDOM_STATE)

def obs_and_perm(tis, model):
    X, y, g, sub = L.tissue_slice(df, mass, tis)
    clf = L.PLSDAClassifier(5) if model == 'PLS-DA' else L.rf()   # rf() = 500 trees (same as observed)
    pipe = Pipeline([('prep', L.SpectraPreproc(0.2)), ('clf', clf)])
    a, _ = L.animal_level_scores(X, y, g, pipe)
    obs = roc_auc_score(a['y'], a['prob']); stat = abs(obs - 0.5)
    lab = pd.Series(y, index=g).groupby(level=0).first(); aids, alab = lab.index.values, lab.values
    hits = 0
    for _ in range(N_PERM):
        m = dict(zip(aids, rng.permutation(alab))); yp = np.array([m[x] for x in g])
        aa, _ = L.animal_level_scores(X, yp, g, pipe)
        hits += abs(roc_auc_score(aa['y'], aa['prob']) - 0.5) >= stat
    return obs, (1 + hits) / (N_PERM + 1)

rows = []
for tis in TISSUES:
    for model in ['PLS-DA', 'RandomForest']:
        obs, p = obs_and_perm(tis, model)
        rows.append(dict(Tissue=tis, Model=model, AUC=round(obs, 3), perm_p=p))
        print(f"{tis:14s} {model:12s} AUC={obs:.3f} perm_p={p:.4f}", flush=True)

res = pd.DataFrame(rows)
res['perm_p_BH'] = multipletests(res['perm_p'], method='fdr_bh')[1]
res.round(4).to_csv('results/permutation_full8.csv', index=False)
print("\n=== 8-test family, BH-adjusted ==="); print(res.round(4).to_string(index=False))
print("\nSaved results/permutation_full8.csv")

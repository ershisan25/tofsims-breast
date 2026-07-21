"""Precision fix for the permutation test: EXHAUSTIVE (exact) permutation for
liver (C(10,6)=210) and serum (C(12,8)=495) — enumerate every animal-label
assignment; higher B (5000) for breast/fur PLS-DA. Same 500-tree RF for
observed and null (no 150/500 mismatch)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import lib_analysis as L

df, mass = L.load()
rng = np.random.default_rng(L.RANDOM_STATE)

def obs_auc(tis, model):
    X, y, g, _ = L.tissue_slice(df, mass, tis)
    clf = L.PLSDAClassifier(5) if model == 'PLS-DA' else L.rf()
    pipe = Pipeline([('prep', L.SpectraPreproc(0.2)), ('clf', clf)])
    a, _ = L.animal_level_scores(X, y, g, pipe)
    return X, y, g, pipe, roc_auc_score(a['y'], a['prob'])

def auc_for_labels(X, g, pipe, animal_labels, aids):
    m = dict(zip(aids, animal_labels)); yp = np.array([m[x] for x in g])
    a, _ = L.animal_level_scores(X, yp, g, pipe)
    return roc_auc_score(a['y'], a['prob'])

rows = []
for tis in ['Breast tissue', 'Fur', 'Liver', 'Blood serum']:
    X, y, g, pipe, obs = obs_auc(tis, 'PLS-DA') if tis in ('Breast tissue','Fur') else (None,)*5
    for model in ['PLS-DA', 'RandomForest']:
        X, y, g, pipe, obs = obs_auc(tis, model)
        stat = abs(obs - 0.5)
        lab = pd.Series(y, index=g).groupby(level=0).first(); aids = lab.index.values; alab = lab.values
        n, n1 = len(alab), int(alab.sum())
        exhaustive = (tis in ('Liver', 'Blood serum'))
        if exhaustive:
            combos = list(itertools.combinations(range(n), n1)); nulls = []
            for c in combos:
                yl = np.zeros(n, int); yl[list(c)] = 1
                nulls.append(abs(auc_for_labels(X, g, pipe, yl, aids) - 0.5))
            nulls = np.array(nulls); p = (nulls >= stat).mean(); B = len(combos); kind = 'exact'
        else:
            # Match the canonical cached table and the manuscript-facing
            # protocol: 5,000 PLS-DA and 1,000 RF null relabellings for the
            # two larger cohorts.  Liver/serum remain exhaustive above.
            B = 5000 if model == 'PLS-DA' else 1000; hits = 0
            for _ in range(B):
                hits += abs(auc_for_labels(X, g, pipe, rng.permutation(alab), aids) - 0.5) >= stat
            p = (1 + hits) / (1 + B); kind = f'MC(B={B})'
        rows.append(dict(Tissue=tis, Model=model, AUC=round(obs, 3), perm_p=round(p, 4), method=kind))
        print(f"{tis:14s} {model:12s} AUC={obs:.3f} p={p:.4f} [{kind}]", flush=True)

res = pd.DataFrame(rows)
res['perm_p_BH'] = multipletests(res.perm_p, method='fdr_bh')[1].round(4)
res.to_csv('results/permutation_exact8.csv', index=False)
print("\n=== canonical 8-test permutation (exact for liver/serum) + BH ===")
print(res.to_string(index=False)); print("\nSaved results/permutation_exact8.csv")

"""Animal-level label-permutation test: is each tissue's discrimination above
chance?  Permutes labels ACROSS ANIMALS (preserving within-animal spot
structure), reruns the full leakage-free leave-one-animal-out pipeline, and
compares observed mouse-level AUC to the null.  p = (1 + #{null>=obs})/(N+1).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lib_analysis as L

df, mass = L.load()
TISSUES = ['Breast tissue', 'Fur', 'Blood serum', 'Liver']
N_PERM = 1000
rng = np.random.default_rng(L.RANDOM_STATE)

def mouse_auc(X, y, groups, clf):
    a, _ = L.animal_level_scores(X, y, groups, clf)
    return roc_auc_score(a['y'], a['prob'])

rows = []
for tis in TISSUES:
    X, y, groups, sub = L.tissue_slice(df, mass, tis)
    # animal -> label table (each animal single-class)
    animals = pd.Series(y, index=groups).groupby(level=0).first()
    aids = animals.index.values; alab = animals.values
    clf = Pipeline([('prep', L.SpectraPreproc(0.2)), ('clf', L.PLSDAClassifier(5))])
    obs = mouse_auc(X, y, groups, clf)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = rng.permutation(alab)                       # permute at ANIMAL level
        lab_map = dict(zip(aids, perm))
        yp = np.array([lab_map[g] for g in groups])
        null[i] = mouse_auc(X, yp, groups, clf)
    p = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    rows.append(dict(Tissue=tis, AUC_obs=obs, null_mean=null.mean(),
                     null_p95=np.percentile(null, 95), p_perm=p))
    print(f"  {tis:14s} obs AUC={obs:.3f}  null mean={null.mean():.3f} "
          f"p95={np.percentile(null,95):.3f}  p_perm={p:.4f}")

out = pd.DataFrame(rows)
out.round(4).to_csv('results/permutation_test.csv', index=False)
print("\nSaved: results/permutation_test.csv")
print("Interpretation: p_perm < 0.05 => tissue discriminates cancer/control above chance.")

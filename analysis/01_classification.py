"""Decisive step: leakage-free, mouse-level classification for all tissues.

Prints a table of mouse-level AUC [95% CI] for PLS-DA and RF, plus cross-tissue
AUC-difference tests (fur vs liver, fur vs serum, liver vs serum).  Also reports
the naive spot-level AUC for contrast.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lib_analysis as L

df, mass = L.load()
TISSUES = ['Breast tissue', 'Fur', 'Blood serum', 'Liver']
MODELS = {'PLS-DA': L.PLSDAClassifier(n_components=5), 'RandomForest': L.rf()}

rows, animal_scores = [], {}
for tis in TISSUES:
    X, y, groups, sub = L.tissue_slice(df, mass, tis)
    n_an = len(np.unique(groups)); n_pos_an = sub.groupby('AnimalID')['Group'].first().eq('Cancer').sum()
    for mname, clf in MODELS.items():
        pipe = Pipeline([('prep', L.SpectraPreproc(min_det=0.2)), ('clf', clf)])
        a, spot_prob = L.animal_level_scores(X, y, groups, pipe)
        pt, lo, hi, _ = L.bootstrap_auc_ci(a)
        spot_auc = roc_auc_score(y, spot_prob)  # naive spot-level for contrast
        animal_scores[(tis, mname)] = a
        rows.append(dict(Tissue=tis, Model=mname, n_animals=n_an,
                         n_cancer_an=int(n_pos_an), n_ctrl_an=int(n_an - n_pos_an),
                         n_spots=len(y), AUC_mouse=pt, CI_lo=lo, CI_hi=hi,
                         AUC_spot_naive=spot_auc))

res = pd.DataFrame(rows)
pd.set_option('display.width', 160, 'display.max_columns', 20)
print("\n===== MOUSE-LEVEL AUC (leakage-free, leave-one-animal-out) vs naive SPOT-level =====")
print(res.round(3).to_string(index=False))
res.round(4).to_csv('results/classification_auc.csv', index=False)

print("\n===== CROSS-TISSUE AUC DIFFERENCE (bootstrap over animals) =====")
diffs = []
for mname in MODELS:
    ad = {t: animal_scores[(t, mname)] for t in TISSUES}
    for ta, tb in [('Fur', 'Liver'), ('Fur', 'Blood serum'), ('Liver', 'Blood serum'),
                   ('Breast tissue', 'Fur')]:
        d, lo, hi, p = L.cross_tissue_diff(ad, ta, tb)
        sig = '' if (lo <= 0 <= hi) else '  *CI excludes 0*'
        diffs.append(dict(Model=mname, comparison=f'{ta} - {tb}', dAUC=d, CI_lo=lo, CI_hi=hi, p=p))
        print(f"  {mname:12s} {ta:13s} - {tb:12s}: dAUC={d:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}{sig}")
pd.DataFrame(diffs).round(4).to_csv('results/cross_tissue_diff.csv', index=False)
print("\nSaved: results/classification_auc.csv, results/cross_tissue_diff.csv")

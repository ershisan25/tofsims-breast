"""Focused response to the 2nd-order audit: is 'fur at chance' robust, or an
artifact of (a) training on spots vs animal-means, (b) the min_det filter?

Grid: train-unit {spot, animal-mean} x model {PLS, RF} x min_det {0,0.1,0.2,0.3}.
AUC = mouse-level LOAO.  perm_p = TWO-SIDED animal-label permutation, stat
|AUC-0.5| (500 perms; lighter RF for the null).  Breast = point AUC only
(obviously significant).  Permutation only for the contested tissues.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import lib_analysis as L

df, mass = L.load()
N_PERM = 500
rng = np.random.default_rng(L.RANDOM_STATE)


def animal_mean(sub):
    agg = sub.groupby('AnimalID').agg(
        y=('Group', lambda s: int((s == 'Cancer').iloc[0])),
        **{f: (f, 'mean') for f in mass}).reset_index()
    return agg[mass].reset_index(drop=True), agg['y'].values, agg['AnimalID'].values


def make(model, md, light=False):
    if model == 'PLS':
        clf = L.PLSDAClassifier(5)
    else:
        clf = RandomForestClassifier(n_estimators=150 if light else 500,
                                     random_state=L.RANDOM_STATE, n_jobs=-1)
    return Pipeline([('prep', L.SpectraPreproc(md)), ('clf', clf)])


def auc_spot(X, y, groups, model, md, light=False):
    a, _ = L.animal_level_scores(X, y, groups, make(model, md, light))
    return roc_auc_score(a['y'], a['prob'])


def auc_animal(Xa, ya, ga, model, md, light=False):
    logo = LeaveOneGroupOut(); prob = np.full(len(ya), np.nan)
    for tr, te in logo.split(Xa, ya, ga):
        est = make(model, md, light); est.fit(Xa.iloc[tr], ya[tr])
        prob[te] = est.predict_proba(Xa.iloc[te])[:, 1]
    return roc_auc_score(ya, prob)


def perm_p(obs, kind, data, model, md):
    stat = abs(obs - 0.5)
    if kind == 'spot':
        X, y, groups = data
        s = pd.Series(y, index=groups).groupby(level=0).first()
        aids, alab = s.index.values, s.values
    else:
        Xa, ya, ga = data; aids, alab = ga, ya
    hits = 0
    for _ in range(N_PERM):
        m = dict(zip(aids, rng.permutation(alab)))
        if kind == 'spot':
            yp = np.array([m[g] for g in groups]); au = auc_spot(X, yp, groups, model, md, light=True)
        else:
            yp = np.array([m[g] for g in ga]); au = auc_animal(Xa, yp, ga, model, md, light=True)
        hits += (abs(au - 0.5) >= stat)
    return (1 + hits) / (N_PERM + 1)


rows = []
for tis in ['Breast tissue', 'Fur', 'Liver', 'Blood serum']:
    X, y, groups, sub = L.tissue_slice(df, mass, tis)
    Xa, ya, ga = animal_mean(sub)
    do_perm = True  # breast p-values now included (Fig S3 request)
    for model in ['PLS', 'RF']:
        for md in [0.0, 0.1, 0.2, 0.3]:
            a_spot = auc_spot(X, y, groups, model, md)
            a_anim = auc_animal(Xa, ya, ga, model, md)
            p_spot = perm_p(a_spot, 'spot', (X, y, groups), model, md) if do_perm else np.nan
            p_anim = perm_p(a_anim, 'animal', (Xa, ya, ga), model, md) if do_perm else np.nan
            rows.append(dict(Tissue=tis, model=model, min_det=md,
                             AUC_spot=round(a_spot, 3), p_spot=round(p_spot, 4) if do_perm else '',
                             AUC_animal=round(a_anim, 3), p_animal=round(p_anim, 4) if do_perm else ''))
            print(f"{tis:14s} {model:3s} md{md:.1f} | spot AUC={a_spot:.3f} p={p_spot if not do_perm else round(p_spot,3)} "
                  f"| animal AUC={a_anim:.3f} p={p_anim if not do_perm else round(p_anim,3)}", flush=True)

res = pd.DataFrame(rows)
res.to_csv('results/sensitivity_grid.csv', index=False)
print("\n=== FUR (contested) ===")
print(res[res.Tissue == 'Fur'].to_string(index=False))
print("\nSaved results/sensitivity_grid.csv")

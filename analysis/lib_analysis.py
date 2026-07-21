"""Shared utilities for the leakage-free, mouse-level analysis.

Fixes addressed:
  A1  mouse-level AUC + bootstrap CI + cross-tissue AUC-difference test
  A2  unit = animal (aggregate held-out spot probs to one score per animal)
  A3  all preprocessing (feature filter, impute floor, log, Pareto) fit INSIDE
      each training fold via an sklearn transformer  -> no leakage
  A4  fold-wise stability selection of the signature ions
  grouping fix: unique physical-animal IDs (Group+Mouse), true leave-one-animal-out
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

import os
META = ['Index', 'Tissue', 'Mouse', 'Spot', 'Group']
RANDOM_STATE = 42
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, 'peak_analysis/all4_50-250/peak_area_table_new.txt')


def load():
    df = pd.read_csv(DATA_PATH, sep='\t')
    # unique physical-animal id: Ca_10b vs Co_10b are different animals
    df['AnimalID'] = df['Group'].str[:2] + '_' + df['Mouse'].astype(str)
    mass = [c for c in df.columns if c not in META and c != 'AnimalID']
    return df, mass


def tissue_slice(df, mass, tissue):
    sub = df[df['Tissue'] == tissue].reset_index(drop=True)
    X = sub[mass].copy()
    y = (sub['Group'] == 'Cancer').astype(int).values
    groups = sub['AnimalID'].values
    return X, y, groups, sub


class SpectraPreproc(BaseEstimator, TransformerMixin):
    """Replicates the notebook transform but fit on TRAIN ONLY (fold-safe).

    fit: keep features with detection >= min_det on train; learn global
         min-nonzero floor and per-feature Pareto mean/std on train (log scale).
    """
    def __init__(self, min_det=0.2):
        self.min_det = min_det

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        det = (X > 0).mean(axis=0)
        self.keep_ = det >= self.min_det
        Xk = X[:, self.keep_]
        nz = Xk[Xk > 0]
        self.floor_ = (nz.min() / 2.0) if nz.size else 1e-9
        Xl = np.log2(np.where(Xk > 0, Xk, self.floor_) + 1.0)
        self.mean_ = Xl.mean(axis=0)
        self.std_ = Xl.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)[:, self.keep_]
        Xl = np.log2(np.where(X > 0, X, self.floor_) + 1.0)
        return (Xl - self.mean_) / np.sqrt(self.std_)


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """Thin PLS-regression-as-classifier wrapper exposing predict_proba."""
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        nc = int(min(self.n_components, X.shape[1], max(1, X.shape[0] - 1)))
        self.model_ = PLSRegression(n_components=nc, scale=False)
        self.model_.fit(X, y.astype(float))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict(X).ravel()
        p = np.clip(p, 0, 1)
        return np.column_stack([1 - p, p])


def animal_level_scores(X, y, groups, estimator):
    """True leave-one-animal-out. Return per-animal (label, mean held-out prob).

    Each fold holds out ALL spots of one physical animal (single class), predicts
    their probs with a model+preproc fit on the other animals only, then averages
    to one score per animal.  => unit of analysis = animal (fixes A2).
    """
    logo = LeaveOneGroupOut()
    spot_prob = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, groups):
        est = _clone_est(estimator)
        est.fit(X.iloc[tr] if hasattr(X, 'iloc') else X[tr], y[tr])
        spot_prob[te] = est.predict_proba(X.iloc[te] if hasattr(X, 'iloc') else X[te])[:, 1]
    d = pd.DataFrame({'AnimalID': groups, 'y': y, 'prob': spot_prob})
    a = d.groupby('AnimalID').agg(y=('y', 'first'), prob=('prob', 'mean')).reset_index()
    return a, spot_prob


def _clone_est(estimator):
    from sklearn.base import clone
    return clone(estimator)


def bootstrap_auc_ci(a, n_boot=2000, seed=RANDOM_STATE):
    """Percentile 95% CI for AUC, resampling ANIMALS stratified by class."""
    rng = np.random.default_rng(seed)
    yv, pv = a['y'].values, a['prob'].values
    idx_pos = np.where(yv == 1)[0]
    idx_neg = np.where(yv == 0)[0]
    point = roc_auc_score(yv, pv)
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(idx_pos, len(idx_pos), replace=True)
        bn = rng.choice(idx_neg, len(idx_neg), replace=True)
        bi = np.concatenate([bp, bn])
        yb, pb = yv[bi], pv[bi]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, np.array(boots)


def cross_tissue_diff(a_dict, ta, tb, n_boot=2000, seed=RANDOM_STATE):
    """Bootstrap AUC(ta)-AUC(tb) over independent animal resamples.

    Returns (diff_point, lo, hi, p_two_sided) where p = 2*min(P(d<0),P(d>0)).
    Unpaired (animals differ across tissues).
    """
    rng = np.random.default_rng(seed)

    def _boot_aucs(a):
        yv, pv = a['y'].values, a['prob'].values
        ip, ineg = np.where(yv == 1)[0], np.where(yv == 0)[0]
        out = []
        for _ in range(n_boot):
            bi = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
            yb = yv[bi]
            if len(np.unique(yb)) < 2:
                out.append(np.nan)
            else:
                out.append(roc_auc_score(yb, pv[bi]))
        return np.array(out)

    da, dbb = _boot_aucs(a_dict[ta]), _boot_aucs(a_dict[tb])
    diff = da - dbb
    diff = diff[~np.isnan(diff)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return diff.mean(), lo, hi, min(p, 1.0)


def rf():
    return RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)

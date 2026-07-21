"""Regenerate main manuscript Figures 2, 4 and 5 from the leakage-free,
animal-level analysis.

Narrative arc (as agreed):
  Fig 2  Mammary tumour carries a robust multi-ion signature that is classified
         with near-perfect, animal-level confidence.
  Fig 4  Extending to the three peripheral tissues: liver and serum are at
         chance; fur is the only promising lead and clearly exceeds liver/serum
         (but is not multiplicity-robust).
  Fig 5  Signature ions are co-detected across tissues (descriptive); no
         peripheral ion reaches animal-level significance.

All classification performance uses leave-one-animal-out (LOAO) with the animal
as the unit.  Score/loading panels (PCA, PLS-DA scores, VIP, RF importance) are
descriptive and fit on all spectra of the tissue for visualisation only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science' styles)

plt.style.use(["science", "no-latex"])
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.stats import mannwhitneyu
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests

import lib_analysis as L
from pathlib import Path


def pipe(clf, min_det=0.2):
    """Canonical leakage-free pipeline: preprocessing fit inside each fold."""
    return Pipeline([("prep", L.SpectraPreproc(min_det)), ("clf", clf)])

FIG = Path(__file__).resolve().parent / "figures"
RES = Path(__file__).resolve().parent / "results"

# Match the 1st author's original palette: Cancer=red, Control=blue; volcano
# red-up / blue-down / grey; bars steelblue; ROC darkorange; YlOrRd heatmaps.
OI = {
    "cancer": "#e74c3c", "control": "#3498db", "up": "#d62728", "down": "#1f77b4",
    "sig": "#d62728", "ns": "#B0B0B0", "bar": "steelblue", "roc": "darkorange",
    "fur": "#e74c3c", "liver": "#3498db", "serum": "#9467bd",
    "pls": "#1f77b4", "rf": "#2ca02c", "black": "#222222", "navy": "navy",
}
HEAT_CMAP = "YlOrRd"
TISSUES = ["Breast tissue", "Fur", "Liver", "Blood serum"]
LABEL = {"Breast tissue": "Mammary", "Fur": "Fur", "Liver": "Liver", "Blood serum": "Serum"}
ROBUST_CORE = [52.02, 53.02, 65.03, 66.02]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
})


# --------------------------------------------------------------------------- #
# Shared computation
# --------------------------------------------------------------------------- #
def animal_ion_stats(sub, mass):
    """Canonical animal-level MWU table for one tissue (matches the SI)."""
    am = sub.groupby("AnimalID").agg(Group=("Group", "first"),
                                     **{m: (m, "mean") for m in mass}).reset_index()
    cancer = am["Group"].eq("Cancer").to_numpy()
    V = am[mass].to_numpy(float)
    p, rbc = [], []
    for j in range(V.shape[1]):
        a, b = V[cancer, j], V[~cancer, j]
        try:
            meth = "exact" if len(a) <= 12 and len(b) <= 12 else "asymptotic"
            r = mannwhitneyu(a, b, alternative="two-sided", method=meth)
            p.append(float(r.pvalue)); rbc.append(2.0 * r.statistic / (len(a) * len(b)) - 1.0)
        except ValueError:
            p.append(1.0); rbc.append(0.0)
    fdr = multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1]
    cm, ctrlm = V[cancer].mean(0), V[~cancer].mean(0)
    n1, n2 = int(cancer.sum()), int((~cancer).sum())
    s1, s2 = V[cancer].std(0, ddof=1), V[~cancer].std(0, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohens_d = np.divide(cm - ctrlm, pooled, out=np.zeros_like(cm), where=pooled > 0)
    return pd.DataFrame({
        "Feature": np.asarray(mass, float),
        "log2fc": np.log2((cm + 1.0) / (ctrlm + 1.0)),
        "rbc": rbc, "p": p, "FDR": fdr, "cohens_d": cohens_d,
        "cancer_mean": cm, "control_mean": ctrlm,
        "sig": fdr < 0.05,
    }), am


def preproc_matrix(X, min_det=0.2):
    """Descriptive preprocessing on all spectra (for score/loading panels)."""
    pp = L.SpectraPreproc(min_det=min_det).fit(X)
    return pp.transform(X), np.asarray(X.columns, float)[pp.keep_]


def vip_scores(Xz, y):
    pls = PLSRegression(n_components=5, scale=False).fit(Xz, y.astype(float))
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = (q[0] ** 2) * (t ** 2).sum(axis=0)
    vip = np.sqrt(w.shape[0] * ((w / np.linalg.norm(w, axis=0)) ** 2 @ ss) / ss.sum())
    return vip, pls


def panel_letter(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=13, pad=8)


# --------------------------------------------------------------------------- #
# Figure 2 — mammary signature (faithful 8-panel A–H, animal-level)
# --------------------------------------------------------------------------- #
def _conf_ellipse(ax, x, y, color, n_std=2.0):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=angle,
                         facecolor=color, edgecolor=color, alpha=0.12, lw=1.3))


def figure2(df, mass):
    sub = df[df.Tissue.eq("Breast tissue")].reset_index(drop=True)
    X, y, groups, _ = L.tissue_slice(df, mass, "Breast tissue")
    stats, am = animal_ion_stats(sub, mass)
    Xz, kept = preproc_matrix(X)
    can_a = am.Group.eq("Cancer").to_numpy()

    fig = plt.figure(figsize=(15.0, 15.5), constrained_layout=True)
    gs = GridSpec(3, 6, figure=fig)

    # A  Volcano: red = enhanced in cancer, blue = reduced, grey = n.s.
    ax = fig.add_subplot(gs[0, 0:2])
    up = stats.sig & (stats.log2fc > 0)
    down = stats.sig & (stats.log2fc <= 0)
    neglogp = -np.log10(stats.p.clip(lower=1e-300))
    ax.scatter(stats.log2fc[~stats.sig], neglogp[~stats.sig], s=26, c=OI["ns"], alpha=0.6, edgecolors="none")
    ax.scatter(stats.log2fc[up], neglogp[up], s=30, c=OI["up"], alpha=0.75, edgecolors="none")
    ax.scatter(stats.log2fc[down], neglogp[down], s=30, c=OI["down"], alpha=0.75, edgecolors="none")
    ax.axhline(-np.log10(0.05), color="black", ls="--", lw=0.9, alpha=0.5)
    ax.axvline(0, color="black", ls="--", lw=0.9, alpha=0.3)
    ax.set(xlabel="Log2 fold change (Cancer / Control)", ylabel="-Log10(p-value)")
    ax.legend(handles=[mpatches.Patch(color=OI["up"], label="Enhanced in Cancer"),
                       mpatches.Patch(color=OI["down"], label="Reduced in Cancer"),
                       mpatches.Patch(color=OI["ns"], label="Not Significant")],
              loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    panel_letter(ax, "A")

    # B  Per-ion violins of the top discriminative ions, original-style labels
    #    (m/z, p, Log2FC, Cohen's d; title coloured by direction; * = signature core)
    top = stats.loc[stats.sig].reindex(
        stats.loc[stats.sig].rbc.abs().sort_values(ascending=False).index).head(10)
    core_mz = {round(m) for m in ROBUST_CORE}
    b_host = fig.add_subplot(gs[0, 2:6]); b_host.axis("off")
    b_host.text(-0.06, 1.07, "B", transform=b_host.transAxes, fontweight="bold", fontsize=12, va="bottom")
    rng = np.random.default_rng(0)
    sub_gs = gs[0, 2:6].subgridspec(2, 5, hspace=0.85, wspace=0.6)
    for i, (_, r) in enumerate(top.iterrows()):
        bax = fig.add_subplot(sub_gs[i // 5, i % 5])
        col = mass[int(np.argmin(np.abs(np.asarray(mass, float) - r.Feature)))]
        v = np.log10(am[col].to_numpy(float) + 1.0)
        for mask, color, pos in [(can_a, OI["cancer"], 0), (~can_a, OI["control"], 1)]:
            vv = v[mask]
            vp = bax.violinplot(vv, positions=[pos], widths=0.8, showmeans=True)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.5); b.set_edgecolor(color)
            for key in ("cbars", "cmins", "cmaxes", "cmeans"):
                vp[key].set_color(color); vp[key].set_linewidth(0.8)
            bax.scatter(pos + (rng.random(len(vv)) - 0.5) * 0.22, vv, s=5,
                        color="black", alpha=0.45, zorder=3, linewidths=0)
        reg_color = OI["up"] if r.log2fc > 0 else OI["down"]
        star = "*" if round(r.Feature) in core_mz else ""
        bax.set_title(f"m/z {r.Feature:.2f}{star}\n$p$={r.p:.0e}\nFC={r.log2fc:.2f}, $d$={r.cohens_d:.1f}",
                      fontsize=6.0, fontweight="bold", color=reg_color, pad=2, linespacing=1.25)
        bax.set_xticks([0, 1]); bax.set_xticklabels(["Ca", "Ct"], fontsize=6.5)
        bax.tick_params(labelsize=6.2)
        if i % 5 == 0:
            bax.set_ylabel("log10(area+1)", fontsize=6.2)

    # C  PCA scores with 95% ellipses
    ax = fig.add_subplot(gs[1, 0:2])
    pcs = PCA(n_components=3, random_state=42).fit(Xz)
    sc = pcs.transform(Xz)
    for lab, color in [("Cancer", OI["cancer"]), ("Control", OI["control"])]:
        m = (sub.Group == lab).to_numpy()
        ax.scatter(sc[m, 0], sc[m, 1], s=26, color=color, alpha=0.6, edgecolors="black", linewidths=0.4, label=lab)
        _conf_ellipse(ax, sc[m, 0], sc[m, 1], color)
    ax.set(xlabel=f"PC1 ({pcs.explained_variance_ratio_[0]*100:.1f}%)",
           ylabel=f"PC2 ({pcs.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    panel_letter(ax, "C")

    # D  PLS-DA scores with ellipses
    ax = fig.add_subplot(gs[1, 2:4])
    vip, pls = vip_scores(Xz, y)
    ts = pls.x_scores_
    for lab, color in [("Cancer", OI["cancer"]), ("Control", OI["control"])]:
        m = (sub.Group == lab).to_numpy()
        ax.scatter(ts[m, 0], ts[m, 1], s=26, color=color, alpha=0.6, edgecolors="black", linewidths=0.4, label=lab)
        _conf_ellipse(ax, ts[m, 0], ts[m, 1], color)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set(xlabel="Latent Variable 1", ylabel="Latent Variable 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    panel_letter(ax, "D")

    # E/H  ROC from the SPOT-level out-of-fold probabilities of the leakage-free
    # LOAO model (192 spectra -> smooth curve; AUC 0.986 = canonical AUC_spot_naive).
    # The animal-level AUC (0.997) is the formal headline reported in Table 1.
    a_pls, sp_pls = L.animal_level_scores(X, y, groups, pipe(L.PLSDAClassifier()))
    a_rf, sp_rf = L.animal_level_scores(X, y, groups, pipe(L.rf()))

    def roc_panel(ax, prob, auc_label, letter, color):
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {auc_label:.3f} (animal-level)")
        ax.plot([0, 1], [0, 1], color=OI["navy"], lw=1.6, ls="--", label="Random classifier")
        ax.text(0.97, 0.20, "curve drawn at spot level", ha="right", fontsize=7.5, style="italic", color="0.4")
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.05),
               xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.grid(alpha=0.25)
        panel_letter(ax, letter)

    roc_panel(fig.add_subplot(gs[1, 4:6]), sp_pls, roc_auc_score(a_pls.y, a_pls.prob), "E", OI["pls"])

    # F  VIP top 15 (steelblue, VIP=1 threshold)
    ax = fig.add_subplot(gs[2, 0:2])
    order = np.argsort(vip)[::-1][:15][::-1]
    ax.barh(range(len(order)), vip[order], color=OI["pls"])
    ax.axvline(1.0, color="red", ls="--", lw=1.6)
    ax.set(yticks=range(len(order)), yticklabels=[f"{kept[i]:.2f}" for i in order],
           xlabel="VIP Score", ylabel="m/z")
    panel_letter(ax, "F")

    # G  RF importance top 15 (steelblue)
    ax = fig.add_subplot(gs[2, 2:4])
    imp = L.rf().fit(Xz, y).feature_importances_
    order = np.argsort(imp)[::-1][:15][::-1]
    ax.barh(range(len(order)), imp[order], color=OI["rf"])
    ax.set(yticks=range(len(order)), yticklabels=[f"{kept[i]:.2f}" for i in order],
           xlabel="Feature Importance", ylabel="m/z")
    panel_letter(ax, "G")

    roc_panel(fig.add_subplot(gs[2, 4:6]), sp_rf, roc_auc_score(a_rf.y, a_rf.prob), "H", OI["rf"])

    fig.savefig(FIG / "fig_2_mammary_signature.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_2_mammary_signature.png", bbox_inches="tight")
    plt.close(fig)
    return {"auc_pls": roc_auc_score(a_pls.y, a_pls.prob), "auc_rf": roc_auc_score(a_rf.y, a_rf.prob),
            "n_sig": int(stats.sig.sum())}


# --------------------------------------------------------------------------- #
# Figure 4 — peripheral tissues only (fur / liver / serum), balanced
# --------------------------------------------------------------------------- #
def figure4(df, mass):
    """15-panel per-tissue figure: serum (A-E), liver (F-J), fur (K-O).
    Columns: PCA | PLS-DA VIP | RF importance | ROC PLS-DA | ROC RF.  (Matches
    the author's Fig 4 caption.)  Row order follows the caption: serum, liver, fur."""
    perm = pd.read_csv(RES / "permutation_exact8.csv")
    row_tissues = ["Blood serum", "Liver", "Fur"]           # caption order
    letters = "ABCDEFGHIJKLMNO"

    fig = plt.figure(figsize=(16.5, 9.8), constrained_layout=True)
    gs = GridSpec(3, 5, figure=fig)

    for r, tissue in enumerate(row_tissues):
        X, y, groups, sub = L.tissue_slice(df, mass, tissue)
        Xz, kept = preproc_matrix(X)
        vip, _ = vip_scores(Xz, y)
        imp = L.rf().fit(Xz, y).feature_importances_
        a_pls, sp_pls = L.animal_level_scores(X, y, groups, pipe(L.PLSDAClassifier()))
        a_rf, sp_rf = L.animal_level_scores(X, y, groups, pipe(L.rf()))
        pp_pls = perm[(perm.Tissue.eq(tissue)) & (perm.Model.eq("PLS-DA"))].perm_p_BH.iloc[0]
        pp_rf = perm[(perm.Tissue.eq(tissue)) & (perm.Model.eq("RandomForest"))].perm_p_BH.iloc[0]

        def ptitle(ax, i, txt):
            ax.set_title(f"{letters[r*5+i]}  {txt}", loc="left", fontweight="bold", fontsize=11, pad=6)

        # col 0 — PCA (no ellipses; red/blue points)
        ax = fig.add_subplot(gs[r, 0]); pcs = PCA(2, random_state=42).fit(Xz); sc = pcs.transform(Xz)
        for lab, color in [("Cancer", OI["cancer"]), ("Control", OI["control"])]:
            m = (sub.Group == lab).to_numpy()
            ax.scatter(sc[m, 0], sc[m, 1], s=18, color=color, alpha=0.65, edgecolors="black", linewidths=0.4, label=lab)
        ax.set(xlabel=f"PC1 ({pcs.explained_variance_ratio_[0]*100:.0f}%)", ylabel=f"PC2 ({pcs.explained_variance_ratio_[1]*100:.0f}%)")
        if r == 0: ax.legend(frameon=False, fontsize=7.5, loc="best")
        ptitle(ax, 0, f"{LABEL[tissue]} — PCA")

        def roc4(ax, a, sp, pbh, color):
            fpr, tpr, _ = roc_curve(y, sp)
            ax.plot(fpr, tpr, color=color, lw=2.0); ax.plot([0, 1], [0, 1], color=OI["navy"], ls="--", lw=1.0)
            ax.text(0.38, 0.10, f"AUC={roc_auc_score(a.y, a.prob):.2f}\nP(BH)={pbh:.2f}", fontsize=8)
            ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="FPR", ylabel="TPR")

        # col 1/2 — PLS-DA: VIP + ROC (blue)
        ax = fig.add_subplot(gs[r, 1]); o = np.argsort(vip)[::-1][:10][::-1]
        ax.barh(range(len(o)), vip[o], color=OI["pls"]); ax.axvline(1.0, color="red", ls="--", lw=1.0)
        ax.set(yticks=range(len(o)), yticklabels=[f"{kept[i]:.2f}" for i in o], xlabel="VIP"); ptitle(ax, 1, "PLS-DA VIP")
        ax = fig.add_subplot(gs[r, 2]); roc4(ax, a_pls, sp_pls, pp_pls, OI["pls"]); ptitle(ax, 2, "PLS-DA ROC")

        # col 3/4 — Random forest: importance + ROC (green)
        ax = fig.add_subplot(gs[r, 3]); o = np.argsort(imp)[::-1][:10][::-1]
        ax.barh(range(len(o)), imp[o], color=OI["rf"])
        ax.set(yticks=range(len(o)), yticklabels=[f"{kept[i]:.2f}" for i in o], xlabel="RF importance"); ptitle(ax, 3, "RF importance")
        ax = fig.add_subplot(gs[r, 4]); roc4(ax, a_rf, sp_rf, pp_rf, OI["rf"]); ptitle(ax, 4, "RF ROC")

    fig.savefig(FIG / "fig_4_peripheral.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_4_peripheral.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5 — shared ions across all four tissues (descriptive co-detection)
# --------------------------------------------------------------------------- #
def figure5(df, mass):
    """Original-style: tissues on rows (Breast/Serum/Liver/Fur), m/z on columns,
    two stacked panels (VIP, RF importance), plasma colourmap, grey where an ion
    is not in that tissue's top set."""
    order = ["Breast tissue", "Blood serum", "Liver", "Fur"]
    row_labels = ["Breast", "Serum", "Liver", "Fur"]
    TOPN = 20
    mass_f = np.asarray(mass, float)
    vip_by, imp_by = {}, {}
    for t in order:
        X, y, _, _ = L.tissue_slice(df, mass, t)
        Xz, kept = preproc_matrix(X)
        vip, _ = vip_scores(Xz, y)
        imp = L.rf().fit(Xz, y).feature_importances_
        vfull = np.full(len(mass), np.nan); ifull = np.full(len(mass), np.nan)
        for j, mz in enumerate(kept):
            idx = int(np.argmin(np.abs(mass_f - mz)))
            vfull[idx] = vip[j]; ifull[idx] = imp[j]
        vip_by[t], imp_by[t] = vfull, ifull

    def top_idx(v):
        valid = np.where(~np.isnan(v))[0]
        return set(valid[np.argsort(v[valid])[::-1][:TOPN]].tolist())

    def build(by):
        tops = {t: top_idx(by[t]) for t in order}
        cols = sorted(set().union(*tops.values()), key=lambda i: mass_f[i])
        M = np.full((len(order), len(cols)), np.nan)
        for r, t in enumerate(order):
            for c, i in enumerate(cols):
                if i in tops[t]:
                    M[r, c] = by[t][i]
        return M, cols

    Mv, cols_v = build(vip_by)
    Mi, cols_i = build(imp_by)

    fig, axes = plt.subplots(2, 1, figsize=(17, 8.6), constrained_layout=True)
    cmap = plt.cm.plasma.copy(); cmap.set_bad("#D9D9D9")
    for ax, M, cols, title, cbar, letter in [
        (axes[0], Mv, cols_v, "VIP Scores Across Samples", "VIP Score", "A"),
        (axes[1], Mi, cols_i, "Random Forest Importance Across Samples", "Random Forest Importance", "B"),
    ]:
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="auto")
        ax.set(xticks=range(len(cols)), yticks=range(len(order)), yticklabels=row_labels)
        ax.set_xticklabels([f"{mass_f[i]:.0f}" for i in cols], rotation=90, fontsize=7)
        ax.set_ylabel("Sample"); ax.set_xlabel("m/z")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
        ax.text(-0.03, 1.12, letter, transform=ax.transAxes, fontweight="bold", fontsize=16, va="top")
        fig.colorbar(im, ax=ax, label=cbar, shrink=0.9, pad=0.01)

    fig.savefig(FIG / "fig_5_shared_ions.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_5_shared_ions.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df, mass = L.load()
    s = figure2(df, mass)
    print("Fig 2:", s)
    figure4(df, mass)
    print("Fig 4: done (fur/liver/serum)")
    figure5(df, mass)
    print("Fig 5: done")


if __name__ == "__main__":
    main()

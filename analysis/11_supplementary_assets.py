"""Build canonical tables and figures for the manuscript/Supplementary Information.

The script deliberately avoids legacy cached p values and recalculates the
animal-level ion tables from the peak-area matrix.  It does not infer a new
result: it renders the already-specified animal-level procedures in a form
suited to publication figures and supporting tables.

Run from this directory with Python 3.10+ and the scientific stack installed:
    MPLCONFIGDIR=/tmp/mpl python3.10 11_supplementary_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "supplementary"
FIGURES = ROOT / "figures"
DATA_PATH = ROOT.parent / "peak_analysis" / "all4_50-250" / "peak_area_table_new.txt"
MANIFEST_PATH = RESULTS / "canonical_manifest.json"

TISSUES = ["Breast tissue", "Fur", "Liver", "Blood serum"]
TISSUE_LABELS = {
    "Breast tissue": "Mammary tissue",
    "Fur": "Fur",
    "Liver": "Liver",
    "Blood serum": "Serum",
}
SLUGS = {
    "Breast tissue": "breast",
    "Fur": "fur",
    "Liver": "liver",
    "Blood serum": "serum",
}
MIN_DET_VALUES = [0.0, 0.1, 0.2, 0.3]
PALETTE = {
    "cancer": "#D55E00",
    "control": "#0072B2",
    "significant": "#009E73",
    "nonsignificant": "#B3B3B3",
    "spot": "#CC79A7",
    "animal": "#009E73",
    "accent": "#E69F00",
    "black": "#222222",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    }
)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Export an SI/main figure in vector and raster formats."""
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def load_input() -> tuple[pd.DataFrame, list[str], dict]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    df = pd.read_csv(DATA_PATH, sep="\t")
    meta = {"Index", "Tissue", "Mouse", "Spot", "Group"}
    mass = [column for column in df.columns if column not in meta]
    df["AnimalID"] = df["Group"].str[:2] + "_" + df["Mouse"].astype(str)
    assert len(mass) == 313, f"Expected 313 aligned features; found {len(mass)}"
    return df, mass, manifest


def animal_means(subset: pd.DataFrame, mass: list[str]) -> pd.DataFrame:
    return (
        subset.groupby("AnimalID")
        .agg(Group=("Group", "first"), **{feature: (feature, "mean") for feature in mass})
        .reset_index()
    )


def mwu_p_value(cancer: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    """Return a two-sided Mann-Whitney p value and signed rank-biserial effect."""
    try:
        method = "exact" if len(cancer) <= 12 and len(control) <= 12 else "asymptotic"
        result = mannwhitneyu(cancer, control, alternative="two-sided", method=method)
        # U is the count of cancer > control pairs plus half ties.
        rbc = 2.0 * result.statistic / (len(cancer) * len(control)) - 1.0
        return float(result.pvalue), float(rbc)
    except ValueError:
        return 1.0, 0.0


def ion_statistics(subset: pd.DataFrame, mass: list[str], tissue: str) -> pd.DataFrame:
    """Recalculate the canonical animal-level ion table for one tissue."""
    aggregated = animal_means(subset, mass)
    cancer_mask = aggregated["Group"].eq("Cancer").to_numpy()
    control_mask = ~cancer_mask
    values = aggregated[mass].to_numpy(dtype=float)
    p_values, rbc = [], []
    for index in range(values.shape[1]):
        p_value, effect = mwu_p_value(values[cancer_mask, index], values[control_mask, index])
        p_values.append(p_value)
        rbc.append(effect)
    fdr = multipletests(np.nan_to_num(p_values, nan=1.0), method="fdr_bh")[1]

    cancer = aggregated.loc[cancer_mask, mass]
    control = aggregated.loc[control_mask, mass]
    cancer_mean = cancer.mean(axis=0).to_numpy()
    control_mean = control.mean(axis=0).to_numpy()
    # The +1 transform is descriptive only; inferential status comes from MWU/FDR.
    log2_fc = np.log2((cancer_mean + 1.0) / (control_mean + 1.0))
    cancer_detect = cancer.gt(0).mean(axis=0).to_numpy()
    control_detect = control.gt(0).mean(axis=0).to_numpy()

    table = pd.DataFrame(
        {
            "Tissue": TISSUE_LABELS[tissue],
            "Feature": np.asarray(mass, dtype=float),
            "n_cancer_animals": int(cancer_mask.sum()),
            "n_control_animals": int(control_mask.sum()),
            "cancer_mean": cancer_mean,
            "control_mean": control_mean,
            "log2_fc_plus1_cancer_control": log2_fc,
            "rank_biserial_cancer_control": rbc,
            "p_mwu": p_values,
            "FDR_BH": fdr,
            "cancer_detection_frequency": cancer_detect,
            "control_detection_frequency": control_detect,
            "absolute_detection_frequency_difference": np.abs(cancer_detect - control_detect),
        }
    )
    table["FDR_significant"] = table["FDR_BH"] < 0.05
    table["detection_difference_gt_0_30"] = (
        table["absolute_detection_frequency_difference"] > 0.30
    )
    return table


def build_ion_tables(
    df: pd.DataFrame, mass: list[str]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    summaries = []
    for tissue in TISSUES:
        table = ion_statistics(df.loc[df["Tissue"].eq(tissue)], mass, tissue)
        tables[tissue] = table
        table.to_csv(OUT / f"ion_statistics_{SLUGS[tissue]}.csv", index=False)
        summaries.append(
            {
                "Tissue": TISSUE_LABELS[tissue],
                "animals": f"{table.n_cancer_animals.iloc[0]}v{table.n_control_animals.iloc[0]}",
                "significant_ions_FDR_lt_0_05": int(table.FDR_significant.sum()),
                "detection_difference_gt_0_30_among_significant": int(
                    (table.FDR_significant & table.detection_difference_gt_0_30).sum()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "ion_summary_canonical.csv", index=False)
    assert int(summary.loc[summary.Tissue.eq("Mammary tissue"), "significant_ions_FDR_lt_0_05"].iloc[0]) == 98
    return tables, summary


def feature_retention_table(df: pd.DataFrame, mass: list[str]) -> pd.DataFrame:
    """Summarise train-fold feature counts for all analysis representations."""
    records: list[dict] = []
    for tissue in TISSUES:
        subset = df.loc[df["Tissue"].eq(tissue)].reset_index(drop=True)
        representations = {
            "spot": (subset[mass].to_numpy(dtype=float), subset["AnimalID"].to_numpy()),
            "animal-mean": (
                animal_means(subset, mass)[mass].to_numpy(dtype=float),
                animal_means(subset, mass)["AnimalID"].to_numpy(),
            ),
        }
        for representation, (matrix, groups) in representations.items():
            for held_out in np.unique(groups):
                training = matrix[groups != held_out]
                detection = (training > 0).mean(axis=0)
                for min_det in MIN_DET_VALUES:
                    records.append(
                        {
                            "Tissue": TISSUE_LABELS[tissue],
                            "training_representation": representation,
                            "held_out_animal": held_out,
                            "min_det": min_det,
                            "features_retained": int((detection >= min_det).sum()),
                        }
                    )
    folds = pd.DataFrame(records)
    summary = (
        folds.groupby(["Tissue", "training_representation", "min_det"], as_index=False)
        .agg(
            n_LOAO_folds=("held_out_animal", "size"),
            retained_min=("features_retained", "min"),
            retained_median=("features_retained", "median"),
            retained_max=("features_retained", "max"),
        )
        .sort_values(["Tissue", "training_representation", "min_det"])
    )
    folds.to_csv(OUT / "feature_retention_by_fold.csv", index=False)
    summary.to_csv(OUT / "feature_retention_summary.csv", index=False)
    return summary


def canonical_primary_table(manifest: dict) -> pd.DataFrame:
    primary = pd.read_csv(RESULTS / manifest["primary_classification"]["file"])
    ci = pd.read_csv(RESULTS / manifest["primary_confidence_intervals"]["file"])
    ci = ci.rename(columns={"AUC_mouse": "AUC_CI_source"})
    keep = [
        "Tissue",
        "Model",
        "n_animals",
        "n_cancer_an",
        "n_ctrl_an",
        "n_spots",
        "AUC_CI_source",
        "CI_lo",
        "CI_hi",
    ]
    merged = primary.merge(ci[keep], on=["Tissue", "Model"], validate="one_to_one")
    assert np.allclose(merged["AUC"], merged["AUC_CI_source"], atol=0.002)
    merged["Sample type"] = merged["Tissue"].map(TISSUE_LABELS)
    merged = merged[
        [
            "Sample type",
            "Tissue",
            "Model",
            "n_animals",
            "n_cancer_an",
            "n_ctrl_an",
            "n_spots",
            "AUC",
            "CI_lo",
            "CI_hi",
            "perm_p",
            "perm_p_BH",
            "method",
        ]
    ]
    merged.to_csv(OUT / "primary_classification_canonical.csv", index=False)
    return merged


def sensitivity_long_table(manifest: dict) -> pd.DataFrame:
    grid = pd.read_csv(RESULTS / manifest["sensitivity"]["file"])
    records = []
    for row in grid.itertuples(index=False):
        for representation, auc_column, p_column in (
            ("spot", "AUC_spot", "p_spot"),
            ("animal-mean", "AUC_animal", "p_animal"),
        ):
            records.append(
                {
                    "Tissue": TISSUE_LABELS[row.Tissue],
                    "training_representation": representation,
                    "Model": "PLS-DA" if row.model == "PLS" else "RandomForest",
                    "min_det": row.min_det,
                    "AUC": getattr(row, auc_column),
                    "raw_permutation_p": getattr(row, p_column),
                    "permutations": 500 if row.Tissue != "Breast tissue" else np.nan,
                    "analysis_role": "exploratory sensitivity analysis",
                }
            )
    long = pd.DataFrame(records).sort_values(
        ["Tissue", "training_representation", "Model", "min_det"]
    )
    long.to_csv(OUT / "sensitivity_grid_long.csv", index=False)
    return long


def stability_table(manifest: dict) -> pd.DataFrame:
    spot = pd.read_csv(RESULTS / manifest["stability"]["spot_file"]).rename(
        columns={
            "RF_selfreq": "RF_fold_selection_frequency",
            "Univ_selfreq": "univariate_fold_selection_frequency",
            "PLS_selfreq": "plsda_fold_selection_frequency",
            "mean_freq": "mean_selection_frequency",
        }
    )
    spot["analysis_level"] = "spot-level"
    spot["detection_rate"] = np.nan

    animal = pd.read_csv(RESULTS / manifest["stability"]["animal_file"]).rename(
        columns={
            "RF_freq": "RF_fold_selection_frequency",
            "Univ_freq": "univariate_fold_selection_frequency",
            "PLS_freq": "plsda_fold_selection_frequency",
            "detect_rate": "detection_rate",
        }
    )
    animal["mean_selection_frequency"] = animal[
        ["RF_fold_selection_frequency", "univariate_fold_selection_frequency"]
    ].mean(axis=1)
    animal["analysis_level"] = "animal-level"

    columns = [
        "analysis_level",
        "Feature",
        "RF_fold_selection_frequency",
        "univariate_fold_selection_frequency",
        "plsda_fold_selection_frequency",
        "mean_selection_frequency",
        "detection_rate",
    ]
    output = pd.concat([spot[columns], animal[columns]], ignore_index=True)
    output["stable_by_both_criteria"] = (
        (output["RF_fold_selection_frequency"] >= 0.9)
        & (output["univariate_fold_selection_frequency"] >= 0.9)
    )
    output.to_csv(OUT / "signature_stability_combined.csv", index=False)
    return output


def transfer_table(manifest: dict) -> pd.DataFrame:
    transfer = pd.read_csv(RESULTS / manifest["transfer"]["file"]).copy()
    transfer["Sample type"] = transfer["Tissue"].map(TISSUE_LABELS)
    transfer["positive_direction"] = transfer["transfer_AUC"] > 0.5
    transfer["BH_significant"] = transfer["perm_p_BH"] < 0.05
    transfer.to_csv(OUT / "transfer_results_canonical.csv", index=False)
    return transfer


def annotate_stability(ion_table: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    breast = ion_table.copy()
    for level, label in (("spot-level", "spot_stable"), ("animal-level", "animal_core")):
        stable = stability.loc[
            (stability.analysis_level == level) & stability.stable_by_both_criteria,
            "Feature",
        ].to_numpy(dtype=float)
        breast[label] = breast.Feature.isin(stable)
    return breast


def plot_main_ion_window(breast: pd.DataFrame, stability: pd.DataFrame) -> None:
    breast = annotate_stability(breast, stability).sort_values("Feature")

    fig, (profile_ax, effect_ax) = plt.subplots(2, 1, figsize=(11.2, 8.0), constrained_layout=True)

    # A — mirror mass spectrum: cancer mean upward, control mean downward (sticks)
    x = breast.Feature.to_numpy()
    cancer_profile = np.log10(breast.cancer_mean.to_numpy() + 1.0)
    control_profile = np.log10(breast.control_mean.to_numpy() + 1.0)
    profile_ax.vlines(x, 0, cancer_profile, color=PALETTE["cancer"], lw=1.1)
    profile_ax.vlines(x, 0, -control_profile, color=PALETTE["control"], lw=1.1)
    profile_ax.axhline(0, color=PALETTE["black"], lw=0.8)
    top = np.nanmax([cancer_profile.max(), control_profile.max()]) * 1.1
    profile_ax.set(xlim=(48, 252), ylim=(-top, top), xlabel="m/z",
                   ylabel="log10(mean peak area + 1)")
    profile_ax.set_yticklabels([f"{abs(t):.1f}" for t in profile_ax.get_yticks()])
    profile_ax.plot([], [], color=PALETTE["cancer"], lw=3, label="Cancer animal mean")
    profile_ax.plot([], [], color=PALETTE["control"], lw=3, label="Control animal mean")
    profile_ax.legend(frameon=False, ncol=2, loc="upper right")
    profile_ax.set_title("A", loc="left", fontweight="bold", fontsize=13)

    # B — animal-level ion effects; stable-signature ions marked (small, no edge)
    colours = np.where(breast.FDR_significant, PALETTE["significant"], PALETTE["nonsignificant"])
    effect_ax.scatter(breast.Feature, breast.rank_biserial_cancer_control,
                      c=colours, s=20, alpha=0.82, edgecolors="none")
    stable_spot = breast.spot_stable
    effect_ax.scatter(breast.loc[stable_spot, "Feature"],
                      breast.loc[stable_spot, "rank_biserial_cancer_control"],
                      facecolors=PALETTE["accent"], edgecolors="none", s=22, zorder=3,
                      label="Stable signature")
    effect_ax.scatter([], [], c=PALETTE["significant"], s=20, label="BH FDR < 0.05")
    effect_ax.scatter([], [], c=PALETTE["nonsignificant"], s=20, label="n.s.")
    effect_ax.axhline(0, color="#555555", lw=0.8)
    effect_ax.set(xlim=(45, 255), ylim=(-1.05, 1.05), xlabel="m/z",
                  ylabel="Rank-biserial effect\n(positive: cancer higher; negative: control higher)")
    effect_ax.set_title("B", loc="left", fontweight="bold", fontsize=13)
    effect_ax.legend(fontsize=7.6, frameon=False, loc="lower right")

    save_figure(fig, "fig_S1_ion_window")


def plot_feature_retention(retention: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), sharey=True, constrained_layout=True)
    for ax, tissue in zip(axes.flat, [TISSUE_LABELS[t] for t in TISSUES]):
        subset = retention.loc[retention.Tissue.eq(tissue)]
        for representation, color in (("spot", PALETTE["spot"]), ("animal-mean", PALETTE["animal"])):
            data = subset.loc[subset.training_representation.eq(representation)].sort_values("min_det")
            ax.fill_between(data.min_det, data.retained_min, data.retained_max, color=color, alpha=0.18)
            ax.plot(data.min_det, data.retained_median, marker="o", color=color, label=representation)
        ax.set(title=tissue, xlabel="Within-training-fold min_det", xlim=(-0.02, 0.32), xticks=MIN_DET_VALUES)
    for ax in axes[:, 0]:
        ax.set_ylabel("Features retained per LOAO training fold")
    axes[0, 0].legend(frameon=False, title="Training representation")
    fig.suptitle("Figure S2. Feature retention under the detection-rate filter", fontsize=13, fontweight="bold")
    save_figure(fig, "fig_S2_feature_retention")


def plot_sensitivity_grid(sensitivity: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    row_order = [
        ("spot", "PLS-DA"),
        ("spot", "RandomForest"),
        ("animal-mean", "PLS-DA"),
        ("animal-mean", "RandomForest"),
    ]
    # Diverging scale centred on chance (AUC 0.5): red = above chance, blue =
    # below chance, white = chance. RdBu_r is colour-blind safe and is the
    # conventional encoding for an AUC grid; it also stops the mammary panel
    # from saturating a sequential map.
    norm = TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=1.0)
    image = None
    for ax, tissue in zip(axes.flat, [TISSUE_LABELS[t] for t in TISSUES]):
        subset = sensitivity.loc[sensitivity.Tissue.eq(tissue)]
        values = np.full((len(row_order), len(MIN_DET_VALUES)), np.nan)
        labels = [["" for _ in MIN_DET_VALUES] for _ in row_order]
        for r, (representation, model) in enumerate(row_order):
            for c, min_det in enumerate(MIN_DET_VALUES):
                row = subset.loc[
                    subset.training_representation.eq(representation)
                    & subset.Model.eq(model)
                    & subset.min_det.eq(min_det)
                ].iloc[0]
                values[r, c] = row.AUC
                labels[r][c] = f"{row.AUC:.2f}\n" + ("p=n/a" if pd.isna(row.raw_permutation_p) else f"p={row.raw_permutation_p:.3f}")
        image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                # White text only on the dark extremes of the diverging map.
                dark = abs(values[r, c] - 0.5) > 0.34
                ax.text(c, r, labels[r][c], ha="center", va="center", color="white" if dark else "black", fontsize=7.4)
        ax.set(
            title=tissue,
            xticks=range(len(MIN_DET_VALUES)),
            xticklabels=[str(value) for value in MIN_DET_VALUES],
            yticks=range(len(row_order)),
            yticklabels=["spot / PLS", "spot / RF", "animal mean / PLS", "animal mean / RF"],
            xlabel="min_det",
        )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.83, label="Mouse-level AUC", ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    colorbar.ax.axhline(0.5, color="#222222", lw=0.8)
    save_figure(fig, "fig_S3_sensitivity_grid")


def plot_stability(stability: pd.DataFrame) -> None:
    # Four separate panels (spot / animal x RF / PLS-DA), each sorted high -> low.
    panels = [
        ("spot-level", "RF_fold_selection_frequency", "#2ca02c", "A  Spot-trained · Random forest"),
        ("spot-level", "plsda_fold_selection_frequency", "#1f77b4", "B  Spot-trained · PLS-DA VIP"),
        ("animal-level", "RF_fold_selection_frequency", "#2ca02c", "C  Animal-level · Random forest"),
        ("animal-level", "plsda_fold_selection_frequency", "#1f77b4", "D  Animal-level · PLS-DA VIP"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), constrained_layout=True)
    for ax, (level, col, color, title) in zip(axes.flat, panels):
        subset = stability.loc[stability.analysis_level.eq(level)].copy()
        subset = subset.sort_values(col, ascending=False).head(15).iloc[::-1]
        y = np.arange(len(subset))
        ax.barh(y, subset[col], height=0.66, color=color)
        ax.axvline(0.9, color=PALETTE["accent"], ls="--", lw=1.2)
        ax.set(yticks=y, yticklabels=[f"m/z {f:.2f}" for f in subset.Feature],
               xlim=(0, 1.05), xlabel="Fold-selection frequency")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    save_figure(fig, "fig_S4_signature_stability")


def plot_transfer(transfer: pd.DataFrame) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.6, 4.3), constrained_layout=True)
    row_order = [("PLS-DA", "Fur"), ("PLS-DA", "Liver"), ("PLS-DA", "Blood serum"), ("RandomForest", "Fur"), ("RandomForest", "Liver"), ("RandomForest", "Blood serum")]
    labels, values, colors, notes = [], [], [], []
    for model, tissue in row_order:
        row = transfer.loc[(transfer.Model.eq(model)) & (transfer.Tissue.eq(tissue))].iloc[0]
        labels.append(f"{model.replace('RandomForest', 'RF')} → {TISSUE_LABELS[tissue]}")
        values.append(row.transfer_AUC)
        colors.append(PALETTE["animal"] if row.positive_direction else PALETTE["spot"])
        notes.append(f"raw p={row.perm_p:.3f};  BH={row.perm_p_BH:.3f}")
    y = np.arange(len(labels))[::-1]
    ax.axvline(0.5, color=PALETTE["black"], ls="--", lw=1.0, zorder=1)
    ax.hlines(y, 0, values, color="#CCCCCC", lw=1.0, zorder=1)
    ax.scatter(values, y, c=colors, s=70, zorder=3, edgecolors="white", linewidths=0.6)
    # Stats reported in a fixed column clear of the markers (max marker AUC ~0.63).
    note_x = 0.80
    for yi, note in zip(y, notes):
        ax.text(note_x, yi, note, va="center", ha="left", fontsize=8.5)
    ax.text(0.5, len(labels) - 0.02, "chance", color=PALETTE["black"], fontsize=8.5, ha="center", va="bottom")
    ax.set(
        xlim=(0, 1.28),
        ylim=(-0.6, len(labels) + 0.15),
        xticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        yticks=y,
        yticklabels=labels,
        xlabel="Transfer AUC of the frozen breast-trained model",
    )
    handles = [
        Line2D([0], [0], marker="o", ls="none", mec="white", mfc=PALETTE["animal"], ms=8, label="AUC > 0.5 (above chance)"),
        Line2D([0], [0], marker="o", ls="none", mec="white", mfc=PALETTE["spot"], ms=8, label="AUC < 0.5 (below chance)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
    ax.set_title("Figure S5. Direct transfer of the breast-trained score to peripheral tissues", loc="left", fontweight="bold")
    save_figure(fig, "fig_S5_transfer")


def write_sample_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tissue in TISSUES:
        subset = df.loc[df.Tissue.eq(tissue)]
        group = subset.groupby("AnimalID").Group.first()
        rows.append(
            {
                "Sample type": TISSUE_LABELS[tissue],
                "Cancer animals": int(group.eq("Cancer").sum()),
                "Control animals": int(group.eq("Control").sum()),
                # `group` contains the group label for each AnimalID.  Its
                # `nunique()` is always two (Cancer/Control), whereas this
                # column must report the number of distinct animals.
                "Total animals": int(group.size),
                "Spectra": int(len(subset)),
                "Spectra per animal (min–max)": f"{subset.groupby('AnimalID').size().min()}–{subset.groupby('AnimalID').size().max()}",
            }
        )
    overview = pd.DataFrame(rows)
    overview.to_csv(OUT / "sample_overview.csv", index=False)
    return overview


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df, mass, manifest = load_input()

    ion_tables, ion_summary = build_ion_tables(df, mass)
    retention = feature_retention_table(df, mass)
    primary = canonical_primary_table(manifest)
    sensitivity = sensitivity_long_table(manifest)
    stability = stability_table(manifest)
    transfer = transfer_table(manifest)
    overview = write_sample_overview(df)

    plot_main_ion_window(ion_tables["Breast tissue"], stability)
    plot_sensitivity_grid(sensitivity)
    plot_stability(stability)
    # Fig S2 (feature retention) and Fig S5 (transfer) figures removed per the
    # story-driven reorganisation; their data files are still written above.

    print("Supplementary assets generated:")
    print(f"  {len(ion_summary)} tissue-level ion summaries")
    print(f"  {len(primary)} primary tissue-by-model rows")
    print(f"  {len(sensitivity)} sensitivity rows")
    print(f"  {len(stability)} stability rows")
    print(f"  {len(transfer)} transfer rows")
    print(f"  {len(overview)} sample-overview rows")


if __name__ == "__main__":
    main()

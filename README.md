# tofsims-breast

Analysis code, result tables, and figures for the study *Machine Learning
Reveals a Mammary Tumor-Associated Multi-Ion Signature and Its Detectability
Across Biological Samples* (Manaprasertsak et al., manuscript in preparation).

## Overview

Using time-of-flight secondary ion mass spectrometry (ToF-SIMS) with
multivariate and machine-learning analysis, we characterise low-mass ion
signatures (< 250 *m/z*) in the MMTV-PyMT mouse model of breast cancer and test
whether a mammary tumor multi-ion signature can be detected in peripheral
samples — mammary gland, fur, liver, and serum from the same animals.

Classification uses the **animal** as the unit of analysis under
**leave-one-animal-out (LOAO)** cross-validation:

- **Unit = the animal.** Held-out spectrum-level scores are averaged to a single
  animal-level score before any AUC or inference is computed.
- **No information leakage.** All preprocessing (within-fold detection filter,
  log2 transform, Pareto scaling) is fit on the training animals inside each
  LOAO fold.

Significance is assessed by animal-label permutation tests with
Benjamini–Hochberg (BH) correction across the eight primary tissue-by-model
tests.

### Key results

| Tissue | LOAO AUC (PLS-DA / RF) | Permutation *P* (BH) |
|---|---|---|
| Mammary | 0.997 / 0.997 | 0.002 / 0.004 |
| Fur | 0.66 / 0.74 | 0.83 / 0.64 |
| Liver | 0.54 / 0.42 | 0.89 |
| Serum | 0.41 / 0.31 | 0.89 |

- A **robust mammary tumor signature** distinguishes tumor-bearing from control
  mice — PLS-DA and Random Forest both reach AUC 0.997 under LOAO, significant by
  animal-label permutation.
- Extending to **peripheral samples** did not reproduce the mammary
  discrimination. Among them, **fur** showed the highest classification
  performance and the greatest overlap in discriminative ions with mammary
  gland, whereas **liver and serum** showed little evidence of discrimination.
- Tumor-associated multi-ion signatures therefore classify mammary tissue
  robustly but are not equally retained across biological samples, with fur
  showing partial retention of the mammary signature.

## Repository structure

```
analysis/
  lib_analysis.py              core library: data loading, animal-ID construction,
                               fold-safe preprocessing (SpectraPreproc), PLS-DA
                               classifier, animal-level LOAO scoring, bootstrap CI
  01_classification.py         LOAO AUC + 2,000-resample bootstrap CI per tissue
  02_permutation.py            animal-label permutation (single model)
  03_univariate_mouselevel.py  mouse-level Mann–Whitney testing + BH FDR
  04_stability.py              spot-level fold-wise stability selection
  05_figures.py                exploratory diagnostic figures
  06_sensitivity.py            sensitivity grid (train-unit × model × min_det)
  07_perm_full.py              full 8-test permutation (all tissue × model)
  08_stability_animal.py       animal-level stability selection
  09_canonical.py              canonical ion analysis + direct-transfer test
  10_permutation_exact.py      exact/Monte-Carlo permutation, canonical 8-test table
  11_supplementary_assets.py   supplementary tables and figures
  make_main_figures.py         main figures (mammary signature, peripheral, shared ions)
  make_ppm_figure.py           mass-calibration accuracy figure
  make_intensity_figure.py     m/z-window rationale figure
  ToFSIMS_analysis.ipynb       documented, end-to-end analysis notebook
  ToFSIMS_main_figures.ipynb   reproducible notebook for the main figures
  original_notebooks/          upstream preprocessing + per-tissue analysis notebooks
  results/                     result tables (CSV) + canonical_manifest.json
  figures/                     main and supplementary figures (PNG)
peak_analysis/all4_50-250/     input-data location (see Data availability)
```

## Requirements

Python 3.10 or higher. Install dependencies:

```bash
pip install -r requirements.txt
```

Core stack: NumPy, pandas, SciPy, scikit-learn, statsmodels, Matplotlib,
SciencePlots.

## Reproducing the analysis

1. Obtain the processed peak tables and place them in
   `peak_analysis/all4_50-250/` (see that directory's README for the expected
   file layout).
2. Run the pipeline from the `analysis/` directory:

   ```bash
   cd analysis
   python 01_classification.py       # LOAO AUC + bootstrap CI    -> results/classification_auc.csv
   python 10_permutation_exact.py    # 8 tissue×model permutation -> results/permutation_exact8.csv
   python 09_canonical.py            # canonical ions + transfer  -> results/preservation_transfer.csv
   python 04_stability.py            # spot-level stability
   python 08_stability_animal.py     # animal-level stability
   python 06_sensitivity.py          # sensitivity grid
   python 11_supplementary_assets.py # supplementary tables + figures
   python make_main_figures.py       # main figures
   ```

   Or open `ToFSIMS_analysis.ipynb` / `ToFSIMS_main_figures.ipynb` for a
   documented, cell-by-cell run.

Random seeds are fixed (`random_state = 42`; 500-tree random forests), so the
canonical AUCs and permutation *p* values are reproducible.

## Data availability

The processed peak tables (< 250 *m/z*) are not distributed in this repository.
They are available from the authors on reasonable request. See
`peak_analysis/all4_50-250/README.md` for the expected file layout.

## Citation

```bibtex
@unpublished{manaprasertsak2026mammary,
  author = {Manaprasertsak, Auraya and Tang, Wei and L{\"o}{\"o}f, Caroline and
            Azemovic, Laila and Hagerling, Catharina and Mohlin, Sofie and
            Pienta, Kenneth J. and Kazi, Julhash U. and Malmberg, Per and
            Hammarlund, Emma U.},
  title  = {Machine Learning Reveals a Mammary Tumor-Associated Multi-Ion
            Signature and Its Detectability Across Biological Samples},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## Acknowledgments

Preprocessing follows the mioXpektron ToF-SIMS toolkit
(https://github.com/kazilab/mioxpektron). Built on the scientific Python
ecosystem: NumPy, SciPy, pandas, scikit-learn, statsmodels, Matplotlib, and
SciencePlots.

## License

MIT — see [LICENSE](LICENSE).

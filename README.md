# tofsims-breast

Analysis code for the study *Integrating ToF-SIMS and machine learning reveals a
mammary tumor-associated multi-ion signature* (Manaprasertsak et al., manuscript
in preparation).

## Overview

Using time-of-flight secondary ion mass spectrometry (ToF-SIMS) with
multivariate and machine-learning analysis, we characterise low-mass ion
signatures (< 250 *m/z*) in the MMTV-PyMT mouse model of breast cancer and test
whether a mammary tumor multi-ion signature can be detected in peripheral
samples — mammary gland, fur, liver, and serum from the same animals.

Classification treats the animal as the experimental unit and is evaluated by
leave-one-animal-out cross-validation: within each fold, preprocessing
(detection filter, log2 transform, Pareto scaling) is fitted on the training
animals and the held-out animal's spectrum scores are averaged to one score per
animal. Significance is assessed by animal-label permutation testing with
Benjamini–Hochberg correction across the eight primary tissue-by-model tests.

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
peak_analysis/all4_50-250/     input-data location (see Data availability)
```

Running the pipeline writes the result tables and figures into `analysis/results/`
and `analysis/figures/`; these outputs are not shipped in the repository.

## Requirements

Python 3.10 or higher. Install dependencies:

```bash
pip install -r requirements.txt
```

Core stack: NumPy, pandas, SciPy, scikit-learn, statsmodels, Matplotlib,
SciencePlots.

## Reproducing the analysis

1. Obtain the processed peak tables (see **Data availability**) and place them in
   `peak_analysis/all4_50-250/`.
2. Run the pipeline from the `analysis/` directory:

   ```bash
   cd analysis
   python 01_classification.py       # LOAO AUC + bootstrap CI
   python 10_permutation_exact.py    # 8 tissue×model permutation
   python 09_canonical.py            # canonical ions + direct transfer
   python 04_stability.py            # spot-level stability
   python 08_stability_animal.py     # animal-level stability
   python 06_sensitivity.py          # sensitivity grid
   python 11_supplementary_assets.py # supplementary tables + figures
   python make_main_figures.py       # main figures
   ```

   Or open `ToFSIMS_analysis.ipynb` / `ToFSIMS_main_figures.ipynb` for a
   documented, cell-by-cell run.

Random seeds are fixed (`random_state = 42`; 500-tree random forests), so the
results are reproducible.

## Data availability

The processed peak tables (< 250 *m/z*) are archived separately on Zenodo:
https://doi.org/10.5281/zenodo.21510506. See `peak_analysis/all4_50-250/README.md`
for the expected file layout.

## Citation

```bibtex
@unpublished{manaprasertsak2026mammary,
  author = {Manaprasertsak, Auraya and Tang, Wei and L{\"o}{\"o}f, Caroline and
            Azemovic, Laila and Mohlin, Sofie and Hagerling, Catharina and
            Pienta, Kenneth J. and Kazi, Julhash U. and Malmberg, Per and
            Hammarlund, Emma U.},
  title  = {Integrating ToF-SIMS and machine learning reveals a mammary
            tumor-associated multi-ion signature},
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

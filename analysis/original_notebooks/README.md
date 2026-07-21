# Original notebooks

Upstream preprocessing and per-tissue analysis notebooks that produced the
aligned peak tables and the original exploratory figures.

| notebook | content |
|---|---|
| `00_preprocessing.ipynb` | recalibration, denoising, baseline correction, normalization, peak detection and alignment (all four tissues, m/z 50–250) |
| `01_pca_area.ipynb` | PCA of peak-area profiles across sample types |
| `02_pca_intensity.ipynb` | PCA of peak-intensity profiles across sample types |
| `03_analysis_all_tissues.ipynb` | combined multi-tissue analysis |
| `04_analysis_breast.ipynb` | mammary-tissue analysis |
| `05_analysis_serum.ipynb` | serum analysis |
| `06_analysis_fur.ipynb` | fur analysis |
| `07_analysis_liver.ipynb` | liver analysis |

These are provided for provenance. The animal-level, leakage-free pipeline in
the parent `analysis/` directory is the analysis reported in the paper.

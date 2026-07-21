# Data placeholder

The analysis code expects three aligned peak tables here (m/z 50–250 window):

| file | description |
|---|---|
| `peak_area_table_new.txt` | tab-delimited peak-**area** table (341 spectra × 313 ions) — the primary input for classification/statistics |
| `peak_intensity_table.csv` | peak-**intensity** table (window-rationale figure) |
| `all_detected_peaks.csv`   | per-spectrum detected-peak list (mass-accuracy figure) |

These processed data files are **not distributed in this repository**. They are
available from the corresponding authors on reasonable request, and will be
deposited in a public archive upon publication. Place the files in this
directory to re-run the pipeline end-to-end.

Columns of `peak_area_table_new.txt`: `Index, Tissue, Mouse, Spot, Group,`
followed by 313 m/z columns. `Group ∈ {Cancer, Control}`; the animal identifier
used for leave-one-animal-out is `Group[:2] + '_' + Mouse`.

[![DOI](https://zenodo.org/badge/1326558795.svg)](https://doi.org/10.5281/zenodo.21858080)

# Evaluating a low-cost transfer-learning approach to deforestation detection in the Western Ghats

This repository contains the code and data for a study that tests whether a ResNet50 classifier fine-tuned on the EuroSAT benchmark (95.68% test accuracy) can detect forest loss in a forest–agriculture landscape of Uttara Kannada, Karnataka, India. Classifier land-cover predictions for Sentinel-2 composites from January–March 2018 and 2024 are compared, tile by tile, with Hansen Global Forest Change tree-cover loss.

The classifier's change predictions agreed with mapped tree-cover loss little better than chance (precision and recall 0.076, MCC 0.021). Hand labels show that it identified only about 42% of the region's forest, and Forest tiles changed label at the same rate with or without mapped loss. A synthetic-clearing experiment shows that tile size would additionally limit even an accurate tile classifier: a single clearing must cover about 30% of a 40 ha tile to change its label half the time, whereas mapped loss patches have a median area of 0.17 ha.

All data sources are public: EuroSAT, Sentinel-2 (via Google Earth Engine), Hansen Global Forest Change (versions 1.11 and 1.13), ESA WorldCover 2021 and SRTM.

## Repository layout

| Path | Contents |
|---|---|
| `EuroSATProject.ipynb` | Original pipeline (submitted version): classifier training, Sentinel-2 acquisition, NDWI correction, change detection, first comparison with Hansen v1.11 |
| `revision/notebooks/revision_experiments.ipynb` | Revision analyses: Cloud Score+ composites, Hansen v1.13 loss-year breakdown, metrics with Wilson intervals, MCC and AUPRC, NDWI threshold sweep, single-date check against WorldCover, synthetic clearings, hand-label and audit packs, five-seed retraining |
| `revision/scripts/score_labels_and_figures.py` | Scores the hand labels and blinded audit; draws the audit and synthetic-clearing figures |
| `revision/scripts/figure2_study_region.py` | Draws the study-region figure |
| `revision/data/` | Per-tile results, metrics, hand labels and audit sheets (listed below) |
| `requirements.txt` | Python environment (Google Colab) |

## Revision data (`revision/data/`)

| File | Description |
|---|---|
| `tiles_csplus.csv`, `tiles_qa60.csv` | One row per tile (6,318): predicted class and confidence for 2018 and 2024, NDWI, valid-pixel share, Hansen loss fraction by year (v1.11 and v1.13), WorldCover fractions, elevation, slope |
| `tiles_csplus_categorised.csv` | Evaluated tiles with their comparison category (TP, FP, FN, TN) |
| `headline_metrics.csv` | Precision, recall, MCC, balanced accuracy and AUPRC for every cloud-mask and reference configuration |
| `seed_variation.csv` | Results for the five retrained classifiers |
| `synthetic_clearing.csv` | Every synthetic-clearing trial (tile, donor, fraction, prediction) |
| `brightness_targets.json` | Fixed EuroSAT brightness statistics used for normalisation |
| `packA_labels.csv`, `packA_key.csv` | Hand labels for 150 random tiles, and the key linking them to tiles |
| `packB_audit.csv`, `audit_key.csv` | Blinded audit of 65 tiles, and the key giving each tile's category |

Exported rasters (`comp_*.tif`, `reference.tif`) and model checkpoints (`*.pth`) are archived on Zenodo with each release rather than stored in this repository.

## Reproducing the revision results

1. Open `revision/notebooks/revision_experiments.ipynb` in Google Colab with a GPU runtime and set your Earth Engine project in the setup cell.
2. Run sections 1–2 and wait for the Earth Engine exports to finish, then run sections 3–12. Outputs are written to `MyDrive/eurosat_revision/`.
3. Section 13 (five-seed retraining) is optional and takes about 35 minutes on a T4 GPU.
4. To re-score the hand labels and audit, or to redraw the figures, run the scripts in `revision/scripts/` with Drive mounted.

The canonical classifier checkpoint (95.68% test accuracy) is required for sections 4 onward; it is included in the Zenodo archive.

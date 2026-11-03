# RSVP-UserBound

Code and data for

> J. Hu, P. Xie, S. Wang, J. Li, and G. Jiang, "User-Bound Limits of Calibration-Free RSVP-EEG Decoding:
> Evidence From Eleven Decoders and 122 Participants," submitted.

The study asks whether the difficulty of calibration-free (leave-one-subject-out, strict domain
generalization) RSVP-EEG decoding belongs to the decoder or to the user. This repository contains

* **PRISM**, the reference decoder, with its training pipeline and the configuration of every PRISM run
  in the paper (main benchmark, training seeds, temporal-block validation, `D` cross, source alignment,
  ablations, random-grouping controls);
* **xDAWN-RG**, the untuned classical pipeline (pyRiemann);
* the **calibration** experiments (PRISM fine-tuned on the user's own data, and a user-only sLDA);
* **preprocessing** of the three public datasets;
* the **analysis scripts** that produce every number, figure, and supplementary table.

The nine comparison deep decoders were implemented following their original publications and, where
available, official code. Their code is not redistributed here, but their participant-level results and
trial predictions are part of the results archive, so every analysis can be reproduced without retraining.

## Installation

Python 3.12 and a CUDA GPU for training (analyses run on the CPU).

```bash
conda env create -f environment.yml
conda activate rsvp-userbound
```

The paper used PyTorch 2.10.0 (CUDA 12.8), NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0, pandas 3.0.1,
MNE-Python 1.12.1, pyRiemann 0.10, torchmetrics 1.8.2, and Matplotlib 3.10.8.

## Data

| Dataset | Participants | Source |
|---|---|---|
| THU | 64 | Zhang et al., *Front. Neurosci.* 2020, <http://bci.med.tsinghua.edu.cn/download.html> |
| CAS | 14 (two sessions) | Zheng et al., *Front. Neurosci.* 2020, <https://doi.org/10.6084/m9.figshare.12824771.v1> |
| THINGS-EEG | 44 of 50 | Grootswagers et al., *Sci. Data* 2022, OpenNeuro [ds003825](https://openneuro.org/datasets/ds003825) |

Preprocess into `data/<DATASET>/Standard_128Hz/sub<k>.npz`
(0.3-40 Hz band-pass, common-average reference, 0-1000 ms epochs with pre-stimulus baseline,
128 Hz, per-trial per-channel z-score):

```bash
python scripts/preprocess_thu_cas.py --dataset THU --raw-dir <folder with S1-S10.mat, ...>
python scripts/preprocess_thu_cas.py --dataset CAS --raw-dir <folder with G1D1.mat, ...>
python scripts/preprocess_things_eeg.py --raw-root <ds003825 root>
```

THINGS-EEG participants 1, 6, 18, 23 (flagged in the source metadata) and 49, 50 (different montage)
are excluded by the configurations. Another data location can be given with `RSVP_DATASET_ROOT`.

*Numerical note.* THU and THINGS-EEG files are reproduced exactly. For CAS (recorded at 1000 Hz),
the 0.3 Hz high-pass of the transfer-function filter is numerically sensitive, and files produced with
other library versions differ from ours by about 1e-3 (z-scored units).

## Results archive and checkpoints

Download from the [release page](../../releases) and extract in the repository root:

* `results.tar.gz` -> `Experiments/`: participant-level results of all runs, trial predictions of the
  eleven decoders, calibration results, and the analysis inputs;
* `prism_checkpoints.tar.gz` -> `Experiments/PRISM/...`: the PRISM model of every LOSO fold on the three
  datasets (needed for re-inference and for the calibration fine-tuning).

## Training PRISM

One configuration per run in `Configs/` (shared defaults in `Configs/config.yaml`):

```bash
python run_model_config.py --config Configs/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8.yaml
```

| Configuration | Paper |
|---|---|
| `prism_thu64_d16_..._lmr_m8`, `prism_cas14_d8_..._lmr_m8`, `prism_things_strict44_d16_..._montagefix` | main benchmark (THU, CAS, THINGS-EEG) |
| `..._trainseed2027`, `..._trainseed2028` | training seeds (THU) |
| `..._valblock` | temporal-block validation control (THU) |
| `prism_thu64_d8_...`, `prism_cas14_d16_...`, `prism_things_strict44_d8_...` | `D` cross |
| `..._sourcebatch_{coral_lam10,dann_lam01,mmd_lam10}` | source alignment |
| `..._ablate_{nows,stage1single,stage2single,bothsingle}`, `..._prewsdrop_010` | ablations |
| `..._randomlmr_m8_s<seed>` | random-grouping controls |

Results are written to `Experiments/PRISM/<DATASET>/cross-subject/<tag>/` (or under `$RSVP_RESULTS_ROOT`).
Trial predictions of the trained folds: `python scripts/positive_class_metrics_reinfer.py --models PRISM`.

## Other experiments

```bash
python scripts/run_xdawn_riemann_loso.py --datasets THU,CAS,THINGS          # xDAWN-RG
python scripts/run_prism_finetune_calibration.py                             # PRISM fine-tuned on the user's data
python scripts/run_within_subject_calibration.py --datasets THU,CAS,THINGS  # user-only sLDA (and xDAWN-RG)
python scripts/profile_prism_compactness.py                                  # parameters, MACs, latency (GPU)
```

## Reproducing the analyses

With the results archive in `Experiments/` and the preprocessed data in `data/`:

```bash
python scripts/analyze_diagnostic_circularity_reliability.py  # descriptors, disjoint trials, reliability, SNR simulation
python scripts/analyze_subject_bound_difficulty.py            # variance decomposition, rankings, gains, regressions
python scripts/multiseed_four_models.py                       # training seeds
python scripts/select_d_by_validation.py                      # choice of D
python scripts/within_vs_cross_subject.py                     # calibration versus calibration-free
python scripts/plot_figures.py                                # figures -> figures/
python scripts/build_supplementary_tables.py                  # supplementary tables -> supplementary_tables/
python scripts/plot_graphical_abstract.py                     # graphical abstract -> figures/
```

Outputs are written to `Experiments/analysis/`. `supplementary_tables/` holds the full tables, of which the Supplementary Material shows a selection. `scripts/compute_erp_topo_bank.py` recomputes the
THU ERP-separation maps of Fig. 4 from the data.

## License

MIT (see `LICENSE`).

# Design Effects for Item Analytics in Clustered Learning Platforms

Reproducibility repository for the paper *How Many Students, or How Many Classrooms? Design Effects for Item Analytics in Clustered Learning Platforms.*

The paper measures, on a large diagnostic mathematics platform, the intraclass correlation (ICC) of item-level correctness within classrooms and its design effect on item statistics, and delivers design-effect-corrected minimum-sample tables for item calibration.

## What is here
- `code/` - the analysis pipeline (Python). The run order is under Reproduce.
- `results/` - all derived, item-level aggregate outputs and figures (committed).
  - `RESULTS.json` - every number of the main analyses, assembled by `results_gate.py`.
  - `revision_results.json` - the three analyses added at revision (class-size sensitivity, cluster-robust coverage by number of classes, corrected discrimination sample-size table), written by `analysis_revision.py`.
  - `item_atlas_t12.csv` - per-item ICC, design effect, response and group counts, subject label, difficulty band.
  - `subject_breakdown.csv`, `difficulty_breakdown.csv`, `ci_ratio_items.csv`, `disc_items.csv`, `min_n_table.csv`, `planner_table.csv`, `assistments_problem_atlas.csv`.
  - `supp_coverage_diagnostics.json`, `supp_small_sample_intervals.json` - additional analyses, not reported in the article (see below).
  - `figures/` - the four paper figures.
- `data/` - where the input files go (they are not redistributed); `data/README.md` explains how to obtain and build them.
- `analysis_plan.md` - the frozen pre-registration (written before the confirmatory runs).
- `requirements.txt`, `LICENSE`.

## Headline numbers (from `results/RESULTS.json` and `results/revision_results.json`)
- Median item ICC (classroom) = 0.132 [0.131, 0.133]; 93.7% of items above 0.05.
- Median design effect on item difficulty = 3.63 (Kish formula) / 2.93 (direct cluster bootstrap; 2.94 Taylor linearization); effective sample about 27% of nominal.
- Naive 95% CI coverage under whole-class sampling about 71% (nominal 95%); cluster-robust 85-91%.
- With exactly C whole classes sampled, cluster-robust coverage (z reference, CR1 variance) is 84.3%, 88.5%, 91.7% and 94.5% at C = 10, 20, 40 and 80.
- Item discrimination design effect only about 1.4.
- Replicates on an independent held-out cohort (ICC 0.100, design effect 2.80).

## Data (not redistributed)
The primary input is the Eedi NeurIPS 2020 Education Challenge data (Wang et al., 2020; arXiv:2007.12061), released by Eedi for research use under terms that forbid row-level redistribution. The second corpus is the public ASSISTments 2009-2010 skill-builder data (Feng, Heffernan, & Koedinger, 2009). This repository therefore ships only derived aggregates and code, not raw responses.

To reproduce from scratch, download the challenge zip and the ASSISTments file into `data/`, then build the answer-level tables with `code/build_inputs.py`. `data/README.md` gives the download links, file names, checksums and row counts.

## Reproduce
Tested with Python 3.12 and the pinned packages in `requirements.txt`. Run every script from `code/`.

```bash
pip install -r requirements.txt
cd code
python build_inputs.py                  # Eedi zip -> data/resp_t12.parquet, data/resp_t34.parquet
python analysis_core.py                 # A1/A3/A8 atlas, H1/H2, frozen item subsamples
python analysis_resample.py             # A2 permutation null, A4 CI ratio, first A5/A7 pass
python analysis_resample2.py            # A5 coverage (500 items, R = 2000), A7 discrimination (1000 items), M4 gap decomposition
python analysis_r2.py                   # M6 i.i.d. coverage baseline, M3 ASSISTments permutation null
python analysis_minN.py                 # A6 corrected minimum-N table + split-half stability
python analysis_robust.py               # A9 robustness, Task 3&4 held-out cohort
python analysis_crossed_latent.py       # M9 classroom-within-quiz ICC, A9b latent logistic ICC
python analysis_assistments.py          # second corpus: ASSISTments classroom ICC and design effect
python analysis_planner.py              # class-based item-tryout planner
python results_gate.py                  # assemble results/RESULTS.json
python figures.py                       # figures
python analysis_revision.py             # RV4 class size, RV7 coverage by number of classes, RV5 discrimination table
python supp_coverage_diagnostics.py     # additional analysis, see below
python supp_small_sample_intervals.py   # additional analysis, see below
```

Global random seed: 20260711. Each script draws from its own stream (the global seed plus a fixed offset), so once its inputs exist a script can be re-run on its own and returns the same numbers. With the pinned packages the full sequence takes about 22 minutes on a desktop PC and regenerates the committed files in `results/` byte for byte. The one exception is the latent logistic ICC median in `RESULTS.json` (`robustness.b_latent_logistic_icc.latent_icc_med`, a numerical optimization), which agrees to six decimal places (0.171376) but can differ in later digits between runs.

What each script reads and writes (paths relative to the repository root):

| Script | Reads | Writes |
|---|---|---|
| `build_inputs.py` | `data/eedi_challenge.zip` | `data/resp_t12.parquet`, `data/resp_t34.parquet` |
| `analysis_core.py` | `data/resp_t12.parquet`, `data/eedi_challenge.zip` (question and subject metadata) | `results/item_atlas_t12.csv`, `subject_breakdown.csv`, `difficulty_breakdown.csv`, `core_headline.json`; `data/subsamples.npz`, `data/student_totals.parquet` |
| `analysis_resample.py` | `data/resp_t12.parquet`, `subsamples.npz`, `student_totals.parquet`, `item_atlas_t12.csv` | `results/resample_results.json`, `ci_ratio_items.csv`, `disc_items.csv` |
| `analysis_resample2.py` | the same, plus `resample_results.json`, `ci_ratio_items.csv` | updates `resample_results.json` (A5, A7, M4); rewrites `disc_items.csv` |
| `analysis_r2.py` | `data/resp_t12.parquet`, `data/assistments09_skill_builder.csv`, `subsamples.npz`, `resample_results.json` | updates `resample_results.json` (M6, M3) |
| `analysis_minN.py` | `data/resp_t12.parquet`, `core_headline.json`, `resample_results.json`, `item_atlas_t12.csv` | `results/min_n_table.csv`, `minN_results.json` |
| `analysis_robust.py` | `data/resp_t12.parquet`, `data/resp_t34.parquet` | `results/robust_results.json` |
| `analysis_crossed_latent.py` | `data/resp_t12.parquet`, `robust_results.json`, `item_atlas_t12.csv` | updates `robust_results.json` (M9, A9b) |
| `analysis_assistments.py` | `data/assistments09_skill_builder.csv` | `results/assistments_problem_atlas.csv`, `assistments_results.json` |
| `analysis_planner.py` | (none) | `results/planner_table.csv`, `planner_results.json` |
| `results_gate.py` | the JSON and CSV files above | `results/RESULTS.json` |
| `figures.py` | `RESULTS.json`, `item_atlas_t12.csv` | `results/figures/*.png` |
| `analysis_revision.py` | `data/resp_t12.parquet`, `subsamples.npz`, `disc_items.csv` | `results/revision_results.json` |
| `supp_coverage_diagnostics.py` | `data/resp_t12.parquet`, `subsamples.npz` | `results/supp_coverage_diagnostics.json` |
| `supp_small_sample_intervals.py` | `data/resp_t12.parquet`, `subsamples.npz` | `results/supp_small_sample_intervals.json` |

The intermediate JSON files (`core_headline.json`, `resample_results.json`, `minN_results.json`, `robust_results.json`, `assistments_results.json`, `planner_results.json`) are regenerated by the scripts and collected into `RESULTS.json`. Input locations can be overridden with the environment variables `EEDI_ZIP`, `EEDI_T12`, `EEDI_T34` and `ASSIST09` (see `data/README.md`). `kill_check.py` is the exploratory check of the clustering structure that was run before the analysis plan was frozen; no reported number depends on it.

## Additional analyses (not reported in the article)
Two scripts extend the coverage experiments of the article. Their results are not part of the article.
- `supp_coverage_diagnostics.py` repeats the budget-stopping coverage experiment (A5, `analysis_resample2.py`) and the fixed-class-count experiment (RV7, `analysis_revision.py`) with the same seeds, so the article's coverages are reproduced exactly, and records per-draw diagnostics: classes per draw; draws that consist of a single class (for these the budget-stopping run uses the binomial SE), with cluster-robust coverage computed with and without them; a t(G-1) reference on the same draws; and the variance without the G/(G-1) factor.
- `supp_small_sample_intervals.py` compares, on fresh fixed-class-count draws (C = 10, 20, 40, 80), the z-reference CR1 interval used in the article with a t(C-1) reference and a wild cluster restricted bootstrap-t (Webb six-point weights, B = 999; MacKinnon, Nielsen, & Webb, 2023, https://doi.org/10.1016/j.jeconom.2022.04.001; Webb, 2023, https://doi.org/10.1111/caje.12661).

## Author

Ismayil Zeynalov, Baku State University, Baku, Azerbaijan.
ORCID: [0009-0003-8067-5904](https://orcid.org/0009-0003-8067-5904).

## Citation

Zeynalov, I. (2026). How many students, or how many classrooms? Design effects for item
analytics in clustered learning platforms. *Measurement: Interdisciplinary Research and
Perspectives*. Advance online publication. https://doi.org/10.1080/15366367.2026.2715676

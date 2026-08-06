# Design Effects for Item Analytics in Clustered Learning Platforms

Reproducibility repository for the paper *How Many Students, or How Many Classrooms? Design Effects for Item Analytics in Clustered Learning Platforms.*

The paper measures, on a large diagnostic mathematics platform, the intraclass correlation (ICC) of item-level correctness within classrooms and its design effect on item statistics, and delivers design-effect-corrected minimum-sample tables for item calibration.

## What is here
- `code/` - the analysis pipeline (Python).
- `results/` - all derived, item-level aggregate outputs and figures (committed). `RESULTS.json` is the single source of truth: every number in the paper is regenerated into it.
  - `item_atlas_t12.csv` - per-item ICC, design effect, response and group counts, subject label, difficulty band.
  - `subject_breakdown.csv`, `difficulty_breakdown.csv`, `ci_ratio_items.csv`, `disc_items.csv`, `min_n_table.csv`.
  - `figures/` - the four paper figures.
- `analysis_plan.md` - the frozen pre-registration (written before the confirmatory runs).
- `requirements.txt`, `LICENSE`.

## Headline numbers (from `results/RESULTS.json`)
- Median item ICC (classroom) = 0.132 [0.131, 0.133]; 93.7% of items above 0.05.
- Median design effect on item difficulty = 3.63 (Kish formula) / 2.93 (direct cluster bootstrap; 2.94 Taylor linearization); effective sample about 27% of nominal.
- Naive 95% CI coverage under whole-class sampling about 71% (nominal 95%); cluster-robust 85-91%.
- Item discrimination design effect only about 1.4.
- Replicates on an independent held-out cohort (ICC 0.100, design effect 2.80).

## Data (not redistributed)
The analysis input is the Eedi NeurIPS 2020 Education Challenge data (Wang et al., 2020; arXiv:2007.12061), released by Eedi for research use under terms that forbid row-level redistribution. This repository therefore ships only derived aggregates and code, not raw responses.

To reproduce from scratch, obtain the challenge data and build an answer-level table `resp_t12.parquet` (Task 1&2) with columns `QuestionId, UserId, IsCorrect, GroupId, QuizId` (join the train file with the answer metadata on `AnswerId`), plus `resp_t34.parquet` for the Task 3&4 replication, and place them in `data/`. See `data/README.md`.

## Reproduce
```bash
pip install -r requirements.txt
cd code
python analysis_core.py       # A1/A3/A8 atlas, H1/H2, subsamples
python analysis_resample.py   # A2 permutation null, A4 CI ratio, A5 coverage, A7 discrimination
python analysis_minN.py       # A6 corrected minimum-N table + split-half stability
python analysis_robust.py     # A9 robustness
python results_gate.py        # assemble results/RESULTS.json (source of truth)
python figures.py             # figures
```
Global random seed: 20260711.

## Author

Ismayil Zeynalov, Baku State University, Baku, Azerbaijan.
ORCID: [0009-0003-8067-5904](https://orcid.org/0009-0003-8067-5904).

## Citation

Zeynalov, I. (2026). How many students, or how many classrooms? Design effects for item
analytics in clustered learning platforms. *Measurement: Interdisciplinary Research and
Perspectives*. Advance online publication. https://doi.org/10.1080/15366367.2026.2715676

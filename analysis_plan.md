# Pre-registered analysis plan (FROZEN before any confirmatory run)

**Paper:** How Many Students, or How Many Classrooms? Design Effects for Item Analytics in Clustered Learning Platforms
**Date frozen:** 2026-07-11
**Author:** I. Zeynalov
**Status:** This file is written BEFORE the confirmatory analysis. The kill-check (`research/kill_check_result.md`) and structure exploration (`scripts/explore.py`) were run first and are declared EXPLORATORY; they informed thresholds below but no confirmatory hypothesis test has been run yet. Any analysis added after this freeze is labelled EXPLORATORY in the manuscript.

## 1. Decision and stakes
A platform test developer / EDM researcher computes item statistics (difficulty = proportion correct; discrimination = item-rest point-biserial) from response logs to decide which items to keep, flag, or retire, and how many responses to collect before an item estimate is trustworthy. Published minimum-sample rules and naive confidence intervals treat responses as i.i.d. Platform responses are nested in classrooms (GroupId). If within-class correlation is non-trivial, the effective sample size is smaller than the raw count, naive CIs under-cover, and i.i.d. minimum-N rules are too lenient. The deliverable is a design-effect (DEFF) corrected minimum-N table plus an ICC/DEFF atlas the practitioner can act on.

## 2. Data (frozen)
- Source: Eedi NeurIPS-2020 Education Challenge, Task 1&2 answer-level join (Wang et al. 2020, arXiv:2007.12061). Local reused copy: `resp_t12.parquet` (columns QuestionId, UserId, IsCorrect, DateAnswered, Confidence, GroupId, QuizId).
- Confirmed structure (exploratory): 15,867,850 responses, 27,613 questions, 11,838 groups, 118,971 students; every (UserId, QuestionId) is unique (no repeat-attempt dependency); within an (item, group) cell responses are from distinct students; QuizId is nested within (item, group) for 98% of cells, so GroupId is the clustering unit.
- Cluster unit: **GroupId** (class group). Response is the observation.
- Task 3&4 (`resp_t34.parquet`, 1.38M responses, the diagnostic image items) used ONLY as an external replication cohort for the ICC/DEFF atlas (A1/A3), not for the primary tests.

## 3. Item inclusion (frozen)
Primary set: items with >= 200 responses AND >= 20 distinct groups AND non-degenerate difficulty (0.02 <= p <= 0.98). Robustness thresholds: {100, 200, 500} responses x {10, 20, 50} groups (A9a).

## 4. Estimands and primary outcomes (frozen)
- ICC_i: one-way random-effects ANOVA intraclass correlation of IsCorrect within GroupId, per item i (linear-probability ICC; Kish/design-effect convention).
- m*_i: burden-weighted average cluster size = sum(m_ig^2)/sum(m_ig) over groups g in item i (Kish, for unequal clusters).
- DEFF_i = 1 + (m*_i - 1) * ICC_i (design effect on item difficulty).
- n_eff_i = n_i / DEFF_i (effective sample size).

**Primary metric 1 = median ICC_i over the primary item set.**
**Primary metric 2 = median DEFF_i over the primary item set.**
**Primary metric 3 = empirical coverage of the naive 95% CI for item difficulty under realistic (whole-class) sampling.**

## 5. Hypotheses and decision rules (frozen, confirmatory)
- H1 (ICC is non-trivial): median ICC_i > 0.05. SUPPORTED if the item-level bootstrap 95% CI (2000 resamples of items) for the median ICC excludes 0.05 from below.
- H2 (design effect matters): median DEFF_i > 1.5. SUPPORTED if the item-level bootstrap 95% CI for the median DEFF excludes 1.5 from below.
- H3 (naive CIs under-cover): under whole-class sampling to a target raw n, the empirical coverage of the naive binomial 95% CI for item difficulty is < 0.95, while the cluster-robust 95% CI covers ~0.95. SUPPORTED if the naive-coverage 95% CI upper bound < 0.95 AND cluster-coverage CI includes 0.95.
- Pre-specified NULL interpretation: if H1/H2 are NOT supported (median ICC <= 0.05 / DEFF <= 1.5), that is reported with full prominence as a certification that i.i.d. item-analytics shortcuts are safe on this platform (a useful negative result), and the paper reframes around the bound. No spin.

## 6. Confirmatory analyses (frozen list)
- **A1 ICC atlas (RQ1/H1):** per-item ANOVA ICC over the primary set; report distribution (median, IQR, deciles); by subject (top-level subject from subject tree) and by difficulty band (p in [.02,.4),[.4,.6),[.6,.8),[.8,.98]). Robustness: latent/logistic random-intercept ICC on a fixed random subsample of 300 items (A9b).
- **A2 Permutation null (certification):** for a fixed random sample of 1000 items, permute GroupId labels within the item (preserving group sizes), B=1000, recompute ICC; report observed vs null 95th pct; fraction of items with observed ICC above null; BH-corrected per-item significance. Certifies ICC reflects real class structure, not unequal-size/estimator artifact.
- **A3 DEFF distribution (RQ2/H2):** per-item DEFF over primary set; median, IQR; by subject; H2 test.
- **A4 CI-width ratio (RQ3):** for a fixed random sample of 1500 items, naive bootstrap 95% CI (resample responses) vs cluster bootstrap 95% CI (resample whole groups), B=2000; report distribution of width ratio (cluster/naive) and its correlation with sqrt(DEFF_ANOVA). Independent resampling validation of the DEFF derivation.
- **A5 Coverage experiment (RQ3/H3, the certification):** for a fixed random sample of 500 items with >= 1500 responses, ground truth tau_i = full-data difficulty. R=2000 repetitions: draw whole groups until >= n responses (n in {100,200,400}); compute naive binomial 95% CI (raw n) and cluster-robust 95% CI (cluster bootstrap over drawn groups); record coverage of tau_i and mean CI width. Report naive vs cluster coverage and width.
- **A6 Minimum-N ladder (RQ4, deliverable):** analytic corrected minimum-N from measured DEFF: for target SE on difficulty in {0.02,0.03,0.05} at reference p in {0.5,0.65,0.8}, i.i.d. N = p(1-p)/SE^2 vs cluster N = DEFF_med * i.i.d. N. Empirical validation: split-half stability curves of the difficulty estimate vs budget n (grid 50..3200), two sampling designs (i.i.d. response split vs disjoint-group cluster split), fixed seed; report the n-gap to reach stability 0.9. Deliver the page-one corrected minimum-N table.
- **A7 Discrimination DEFF (RQ5):** discrimination_i = item-rest point-biserial (correlation of IsCorrect_i with the student's leave-item-out mean correctness), for items in the primary set whose respondents have >= 5 other answered items. DEFF_disc via cluster vs naive bootstrap ratio on a fixed sample of 1000 items. Honest report of whether discrimination is more/less/equally affected than difficulty (possible partial null).
- **A8 Heterogeneity:** ICC & DEFF by top-level subject and difficulty band (from A1); identify where DEFF is largest/smallest.
- **A9 Robustness:** (a) inclusion thresholds; (b) linear vs logistic ICC subsample; (c) GroupId vs QuizId cluster; (d) cap on max cluster size; (e) min group size >= 2; (f) Task 3&4 replication of the atlas.
- **A10 Multiplicity:** item-level bootstrap CIs for the H1/H2 medians; BH correction for per-item permutation significance (A2). Effect sizes reported as the medians and DEFF magnitudes themselves.

## 7. Reviewer-hot-button pre-emption map
1. "Linear ICC on binary data is wrong" -> primary follows Kish DEFF convention (linear-probability ICC is the correct input to DEFF); logistic/latent ICC robustness (A9b) shows substantiveness holds.
2. "ICC is an artifact of unequal cluster sizes / small samples" -> permutation null A2 certifies it.
3. "Cherry-picked high-volume items" -> primary set = all items >=200 resp & >=20 groups (15k+), not the top items; kill-check top-30 disclosed as exploratory.
4. "DEFF math is just Kish, not new" -> we do not claim the formula; the contribution is the empirical ICC/DEFF atlas for platform item analytics + the corrected minimum-N tables + the coverage certification. A4/A5 provide model-free resampling validation.
5. "Multilevel IRT already handles this" -> addressed in Related Work + Discussion: multilevel IRT is model-based and rarely used by practitioners doing CTT-style item screening; we quantify the error of the shortcut they actually use and give a one-number correction. (Finalize against novelty-gate refs.)
6. "Only one platform" -> Task 3&4 internal replication (A9f) + honest external-validity limitation; the method transfers, the ICC magnitude is platform-specific.
7. "No significance / effect sizes" -> bootstrap CIs, permutation null, BH correction, coverage experiment.

## 8. Leakage / independence protocol
Descriptive/measurement study; no predictive train/test leakage surface. The only split is the split-half stability (A6): halves are DISJOINT in groups for the cluster arm (group-aware) and in responses for the i.i.d. arm; a student answers each item once so per-item halves are clean. Full-data difficulty (A5 ground truth) is computed once and touched only for coverage scoring.

## 9. Seeds, compute, provenance
- Global seed 20260711. Bootstrap B=2000 (CI), permutation B=1000, coverage R=2000. Fixed random item subsamples drawn once under the seed and logged.
- Bootstrap-heavy analyses (A2/A4/A5/A7) run on pre-registered fixed random item subsamples (sizes above) to bound compute; ANOVA-based A1/A3 run on ALL included items.
- `data/PROVENANCE.md` records source, version, access, license, row/col counts, SHA-256.
- Data license (Eedi CC BY-NC-ND style): NO row-level redistribution; the public repo ships DERIVED aggregate statistics (per-item ICC/DEFF/n) and code only.

## 10. Results Gate (blocks writing)
Every primary metric with a bootstrap 95% CI; H1/H2/H3 decisions logged by the results gate; permutation-null certification reported; robustness deltas reported; all numbers regenerated by a single `results_gate.py` into one `RESULTS.json` (recompute, never retype). Negative results reported in full.

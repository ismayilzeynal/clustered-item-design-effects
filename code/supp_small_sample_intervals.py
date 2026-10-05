# -*- coding: utf-8 -*-
"""Additional analysis (not reported in the article): small-sample interval comparison.

Fixed-class-count design as in analysis_revision.py (RV7): exactly C whole classes,
C = 10, 20, 40, 80, drawn without replacement, on the same 500-item coverage subsample,
with fresh draws (R = 200 per item) and a separate seed. Three 95% intervals for item
difficulty are evaluated on the same draws:
  - z reference with the CR1 cluster-robust variance (the interval used in the article)
  - t(C-1) reference with the same variance
  - wild cluster restricted (WCR) bootstrap-t with Webb six-point weights, B = 999, test
    inverted at the true item difficulty (MacKinnon, Nielsen & Webb, 2023; Webb, 2023).
    One B x C weight matrix is drawn per item and used for all R draws of that item.

Inputs:  EEDI_T12 (default ../data/resp_t12.parquet; see data/README.md),
         ../data/subsamples.npz (written by analysis_core.py).
Seed:    20260711 + 900.
Output:  ../results/supp_small_sample_intervals.json
Run:     from code/:  python supp_small_sample_intervals.py
"""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

SEED = 20260711 + 900
SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
OUT = "../results"
Z = 1.959964
R = 200
B = 999
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])

ss = np.load("../data/subsamples.npz")
df = pq.read_table(SRC, columns=["QuestionId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32); df["GroupId"] = df["GroupId"].astype(np.int32)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)
ids = ss["coverage"]
sub = df[df["QuestionId"].isin(set(ids.tolist()))]
items = {}
for q, g in sub.groupby("QuestionId"):
    y = g["IsCorrect"].values.astype(np.float64)
    _, inv = np.unique(g["GroupId"].values, return_inverse=True)
    items[int(q)] = (np.bincount(inv).astype(np.float64), np.bincount(inv, weights=y), float(y.mean()))

rng = np.random.default_rng(SEED)
out = {"note": "Additional analysis, not reported in the article.", "seed": SEED, "R_per_item": R, "B": B,
       "weights": "Webb six-point", "n_items": int(len(ids))}
for C in [10, 20, 40, 80]:
    tq = stats.t.ppf(0.975, C - 1)
    cz = []; ct = []; cw = []
    for q in ids:
        counts, sums, tau = items[int(q)]
        k = len(counts)
        if k < C: continue
        picks = np.stack([rng.permutation(k)[:C] for _ in range(R)])
        mm = counts[picks]; sg = sums[picks]
        N = mm.sum(1); ph = sg.sum(1) / N
        se = np.sqrt((C / (C - 1)) * np.sum((sg - mm * ph[:, None]) ** 2, 1)) / N
        cz.append(np.abs(ph - tau) <= Z * se)
        ct.append(np.abs(ph - tau) <= tq * se)
        tobs = np.abs(ph - tau) / np.where(se > 0, se, np.nan)
        u = sg - mm * tau
        W = WEBB[rng.integers(0, 6, size=(B, C))]
        cov_w = np.empty(R, bool)
        for r in range(R):
            wu = W * u[r]
            phs = tau + wu.sum(1) / N[r]
            res_g = mm[r] * tau + wu - np.outer(phs, mm[r])
            ses = np.sqrt((C / (C - 1)) * np.sum(res_g ** 2, 1)) / N[r]
            ts = np.abs(phs - tau) / np.where(ses > 0, ses, np.nan)
            if not np.isfinite(tobs[r]):
                cov_w[r] = False if ph[r] != tau else True
                continue
            cov_w[r] = np.mean(ts >= tobs[r]) > 0.05
        cw.append(cov_w)
    cz = np.concatenate(cz); ct = np.concatenate(ct); cw = np.concatenate(cw)
    out[str(C)] = {"n_draws": int(len(cz)), "z_CR1": float(cz.mean()), "t_CR1": float(ct.mean()),
                   "t_crit": float(tq), "WCR_webb": float(cw.mean())}
    print(C, json.dumps(out[str(C)]), flush=True)
with open(f"{OUT}/supp_small_sample_intervals.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE supp_small_sample_intervals")

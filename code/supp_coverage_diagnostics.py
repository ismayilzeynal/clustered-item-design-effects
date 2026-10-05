# -*- coding: utf-8 -*-
"""Additional analysis (not reported in the article): coverage diagnostics on the article's draws.

Repeats the two coverage experiments of the article with the same seeds and the same
random-number call sequence, so the article's coverages are reproduced exactly, and records
further quantities for every draw:
  A5   whole classes drawn in random order until the response budget (100, 200, 400) is
       reached, as in analysis_resample2.run_coverage (R = 2000 draws per item, 500 items,
       seed 20260711 + 5): classes per draw; draws that consist of a single class (for these
       run_coverage uses the binomial SE); cluster-robust coverage with and without those
       draws; coverage with a t(G-1) reference on the draws with two or more classes.
  RV7  exactly C whole classes, C = 10, 20, 40, 80, as in analysis_revision.py (R = 800 draws
       per item, 500 items, seed 20260711 + 70): coverage with z and the CR1 variance (the
       article's interval), with t(C-1) and CR1, and with z and no C/(C-1) factor (CR0).
Also records the number of classes available per item in the coverage subsample.

Inputs:  EEDI_T12 (default ../data/resp_t12.parquet; see data/README.md),
         ../data/subsamples.npz (written by analysis_core.py).
Seed:    20260711 (+5 for A5, +70 for RV7).
Output:  ../results/supp_coverage_diagnostics.json
Run:     from code/:  python supp_coverage_diagnostics.py
"""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

SEED = 20260711
SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
OUT = "../results"
Z = 1.959964
TQ = np.concatenate([[np.nan], stats.t.ppf(0.975, np.arange(1, 5001))])

ss = np.load("../data/subsamples.npz")
df = pq.read_table(SRC, columns=["QuestionId", "UserId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32); df["GroupId"] = df["GroupId"].astype(np.int32)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)

ids = ss["coverage"]
sub_all = df[df["QuestionId"].isin(set(ids.tolist()))]
grp = {q: g for q, g in sub_all.groupby("QuestionId")}
items = {}
for q in ids:
    g = grp[int(q)]
    y = g["IsCorrect"].values.astype(np.float64); gid = g["GroupId"].values
    uniq, inv = np.unique(gid, return_inverse=True)
    items[int(q)] = (np.bincount(inv).astype(np.float64), np.bincount(inv, weights=y), float(y.mean()))

kpool = np.array([len(items[int(q)][0]) for q in ids])
res = {"note": ("Additional analysis, not reported in the article. Same seeds and draw sequence as "
                "analysis_resample2.py (A5) and analysis_revision.py (RV7); the article's coverages "
                "are reproduced exactly."),
       "n_items": int(len(ids)),
       "classes_per_item": {"min": int(kpool.min()), "median": float(np.median(kpool)), "max": int(kpool.max())}}
print("classes per item", json.dumps(res["classes_per_item"]), flush=True)

rng = np.random.default_rng(SEED + 5)
a5 = {}
for n_target in (100, 200, 400):
    cn = []; cc = []; kd = []; ctt = []; one = []; per_item_one = []
    for q in ids:
        counts, sums, tau = items[int(q)]
        k = len(counts)
        n_one_item = 0
        for r in range(2000):
            perm = rng.permutation(k)
            cs = np.cumsum(counts[perm])
            j = int(np.searchsorted(cs, n_target, side="left"))
            if j >= k: j = k - 1
            kk = j + 1
            pk = perm[:kk]
            mm = counts[pk]; sg = sums[pk]
            nn = mm.sum(); ph = sg.sum() / nn
            se_n = np.sqrt(ph * (1 - ph) / nn) if 0 < ph < 1 else 0.0
            cn.append(abs(ph - tau) <= Z * se_n)
            if kk > 1:
                se_c = np.sqrt((kk / (kk - 1)) * np.sum((sg - mm * ph) ** 2)) / nn
                ctt.append(abs(ph - tau) <= TQ[kk - 1] * se_c)
            else:
                se_c = se_n
                ctt.append(np.nan)
                n_one_item += 1
            cc.append(abs(ph - tau) <= Z * se_c)
            kd.append(kk); one.append(kk == 1)
        per_item_one.append(n_one_item)
    cn = np.array(cn); cc = np.array(cc); kd = np.array(kd); one = np.array(one); ctt = np.array(ctt, dtype=float)
    pio = np.array(per_item_one)
    a5[str(n_target)] = {
        "naive_coverage": float(cn.mean()), "cluster_coverage": float(cc.mean()),
        "mean_clusters_drawn": float(kd.mean()),
        "n_draws": int(len(cc)), "one_class_draws": int(one.sum()), "one_class_frac": float(one.mean()),
        "items_with_any_one_class_draw": int((pio > 0).sum()),
        "max_one_class_frac_single_item": float(pio.max() / 2000),
        "cluster_coverage_excl_one_class": float(cc[~one].mean()),
        "cluster_coverage_among_one_class": (float(cc[one].mean()) if one.any() else None),
        "t_ref_coverage_excl_one_class": float(np.nanmean(ctt)),
        "draws_with_2_to_5_classes": int(((kd >= 2) & (kd <= 5)).sum()),
        "p10_p50_p90_clusters": [float(x) for x in np.percentile(kd, [10, 50, 90])],
    }
    print("budget", n_target, json.dumps(a5[str(n_target)]), flush=True)
res["budget_stopping_A5"] = a5

rng3 = np.random.default_rng(SEED + 70)
rv7 = {}
for C in [10, 20, 40, 80]:
    cn = []; cc = []; ct = []; c0 = []
    tq = stats.t.ppf(0.975, C - 1)
    for q in ids:
        counts, sums, tau = items[int(q)]
        k = len(counts)
        if k < C: continue
        for r in range(800):
            pick = rng3.permutation(k)[:C]
            mm = counts[pick]; sg = sums[pick]
            nn = mm.sum(); ph = sg.sum() / nn
            se_n = np.sqrt(ph * (1 - ph) / nn) if 0 < ph < 1 else 0.0
            cn.append(abs(ph - tau) <= Z * se_n)
            se_c = np.sqrt((C / (C - 1)) * np.sum((sg - mm * ph) ** 2)) / nn
            cc.append(abs(ph - tau) <= Z * se_c)
            ct.append(abs(ph - tau) <= tq * se_c)
            se_0 = np.sqrt(np.sum((sg - mm * ph) ** 2)) / nn
            c0.append(abs(ph - tau) <= Z * se_0)
    rv7[str(C)] = {"naive_coverage": float(np.mean(cn)), "cluster_coverage_z_CR1": float(np.mean(cc)),
                   "cluster_coverage_t_CR1": float(np.mean(ct)), "t_crit": float(tq),
                   "cluster_coverage_z_CR0": float(np.mean(c0)), "n_draws": len(cc)}
    print("fixed_C", C, json.dumps(rv7[str(C)]), flush=True)
res["fixed_class_count_RV7"] = rv7

with open(f"{OUT}/supp_coverage_diagnostics.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE supp_coverage_diagnostics")

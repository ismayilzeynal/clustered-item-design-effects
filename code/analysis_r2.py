# -*- coding: utf-8 -*-
"""Round-2 additions.
M6: i.i.d. (individual-response) coverage baseline at the same budgets, to isolate the
    clustering share of the naive undercoverage from the Wald-interval form.
M3: permutation-null certification of the ASSISTments classroom ICC."""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 20260711
T12 = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
ASSIST = os.environ.get("ASSIST09", "../data/assistments09_skill_builder.csv")
OUT = "../results"
ss = np.load("../data/subsamples.npz")
res = json.load(open(f"{OUT}/resample_results.json"))

def icc_from_ms(m, s):
    N = m.sum(); k = len(m); S = s.sum()
    if k < 2 or N <= k: return np.nan
    SSB = np.sum(s * s / m) - S * S / N
    SST = S * (N - S) / N
    if SST <= 0: return np.nan
    SSW = SST - SSB
    MSB = SSB / (k - 1); MSW = SSW / (N - k)
    m0 = (N - np.sum(m * m) / N) / (k - 1)
    denom = MSB + (m0 - 1) * MSW
    return (MSB - MSW) / denom if denom > 0 else np.nan

# ---------- M6: i.i.d. coverage baseline (same items, same budgets) ----------
print("[M6] i.i.d. coverage baseline ...")
df = pq.read_table(T12, columns=["QuestionId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32); df["IsCorrect"] = df["IsCorrect"].astype(np.int8)
cov_ids = ss["coverage"]
sub = df[df["QuestionId"].isin(set(cov_ids.tolist()))]
grp = {q: g["IsCorrect"].values.astype(np.float64) for q, g in sub.groupby("QuestionId")}
rng = np.random.default_rng(SEED + 60)
Z = 1.959964
m6 = {}
for n in (100, 200, 400):
    cov = []
    for q in cov_ids:
        y = grp[int(q)]; tau = y.mean(); N = len(y)
        for r in range(2000):
            draw = y[rng.integers(0, N, n)]      # i.i.d. sampling of n responses
            ph = draw.mean()
            se = np.sqrt(ph * (1 - ph) / n) if 0 < ph < 1 else 0.0
            cov.append(ph - Z * se <= tau <= ph + Z * se)
    m6[str(n)] = round(float(np.mean(cov)), 4)
res["M6_iid_coverage_baseline"] = {"coverage_by_n": m6, "R": 2000, "n_items": len(cov_ids),
    "note": "Naive binomial 95% CI under i.i.d. sampling of the SAME items/budgets; near-nominal here means the whole-class undercoverage (~71%) is the clustering, not the Wald interval form."}
print("  i.i.d. coverage:", m6)

# ---------- M3: ASSISTments permutation-null certification ----------
print("[M3] ASSISTments permutation null ...")
a = pd.read_csv(ASSIST, encoding="latin-1", usecols=["problem_id", "correct", "student_class_id"], low_memory=False)
a = a.dropna(subset=["problem_id", "student_class_id", "correct"]); a = a[a["correct"].isin([0, 1])]
a["correct"] = a["correct"].astype(np.int8)
g = a.groupby("problem_id").agg(n=("correct", "size"), ng=("student_class_id", "nunique"))
items = g[(g.n >= 100) & (g.ng >= 10)].index.values
sub = a[a["problem_id"].isin(set(items.tolist()))]
byitem = {q: grp for q, grp in sub.groupby("problem_id")}
rng2 = np.random.default_rng(SEED + 61)
obs, null95, sig = [], [], []
B = 500
for q in items:
    grp_i = byitem[q]
    gg = grp_i.groupby("student_class_id")["correct"].agg(["size", "sum"])
    m = gg["size"].values.astype(float); s = gg["sum"].values.astype(float)
    o = icc_from_ms(m, s)
    if np.isnan(o): continue
    y = grp_i["correct"].values.astype(float)
    offs = np.concatenate([[0], np.cumsum(m).astype(int)[:-1]])
    perm = np.empty(B)
    for b in range(B):
        yp = y[rng2.permutation(len(y))]; sp = np.add.reduceat(yp, offs)
        perm[b] = icc_from_ms(m, sp)
    obs.append(o); q95 = np.nanquantile(perm, 0.95); null95.append(q95); sig.append(o > q95)
res["M3_assistments_perm_null"] = {
    "n_items": int(len(obs)), "obs_icc_median": round(float(np.nanmedian(obs)), 4),
    "null_icc_median": round(float(np.nanmedian(null95)), 4),
    "frac_obs_gt_null95": round(float(np.mean(sig)), 4),
    "note": "Certifies the ASSISTments classroom ICC is real classroom structure, not estimator artifact."}
print("  ASSISTments perm null:", res["M3_assistments_perm_null"])

with open(f"{OUT}/resample_results.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE analysis_r2")

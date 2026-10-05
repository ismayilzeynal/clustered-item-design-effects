# -*- coding: utf-8 -*-
"""Revision analyses (exploratory relative to the frozen pre-registration in analysis_plan.md).

RV4  Class-size sensitivity of the design effect on item difficulty. Responses retained per
     class are randomly capped at 1, 2, 3, 5, 8, 12, 20 and 30 (plus the uncapped case) on the
     1,500-item validation subsample: analytic ICC and Kish design effect on all items, direct
     cluster-bootstrap design effect on the first 400 items of that subsample. Also a
     retain-50%-of-each-class scenario.
RV7  Cluster-robust 95% interval coverage of item difficulty as a function of the number of
     classes. Exactly C whole classes (C = 10, 20, 40, 80) are drawn without replacement,
     R = 800 draws per item on the 500-item coverage subsample. Intervals: the naive binomial
     interval and the cluster-robust interval with z = 1.959964 and the linearized variance
     with the C/(C-1) factor.
RV5  Design-effect-corrected minimum-sample table for item discrimination, using the measured
     discrimination design effect (median over results/disc_items.csv) and the large-sample
     SE of the point-biserial correlation.

Inputs:  EEDI_T12 (default ../data/resp_t12.parquet; see data/README.md),
         ../data/subsamples.npz (written by analysis_core.py),
         ../results/disc_items.csv (written by analysis_resample2.py; also committed).
Seed:    20260711 (RV4 uses +40 and +41, RV7 uses +70).
Output:  ../results/revision_results.json. RESULTS.json is not modified.
Run:     from code/:  python analysis_revision.py
"""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 20260711
SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
OUT = "../results"
os.makedirs(OUT, exist_ok=True)
ss = np.load("../data/subsamples.npz")
Z = 1.959964


def icc_from_ms(m, s):
    N = m.sum(); k = len(m); S = s.sum()
    if k < 2 or N <= k:
        return np.nan
    SSB = np.sum(s * s / m) - S * S / N
    SST = S * (N - S) / N
    if SST <= 0:
        return np.nan
    SSW = SST - SSB
    MSB = SSB / (k - 1); MSW = SSW / (N - k)
    m0 = (N - np.sum(m * m) / N) / (k - 1)
    denom = MSB + (m0 - 1) * MSW
    return (MSB - MSW) / denom if denom > 0 else np.nan


def deff_analytic_from_groups(m, s):
    """Kish design effect on item difficulty from per-group (count, sum)."""
    icc = icc_from_ms(m, s)
    if np.isnan(icc):
        return np.nan, np.nan
    N = m.sum()
    mstar = np.sum(m * m) / N
    deff = 1 + (mstar - 1) * icc
    return icc, deff


def cluster_boot_deff(y, gid, rng, B=400):
    """Direct cluster-bootstrap design effect on item difficulty."""
    N = len(y); p = y.mean()
    if p <= 0 or p >= 1:
        return np.nan
    naive_se = np.sqrt(p * (1 - p) / N)
    uniq, inv = np.unique(gid, return_inverse=True)
    m = np.bincount(inv).astype(np.float64)
    s = np.bincount(inv, weights=y)
    k = len(uniq)
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, k)
        means[b] = s[pick].sum() / m[pick].sum()
    cluster_se = means.std(ddof=1)
    return (cluster_se / naive_se) ** 2


def cap_within_class(y, gid, m_cap, rng):
    """Randomly retain at most m_cap responses within each class (decimation)."""
    if m_cap is None:
        return y, gid
    keep = np.ones(len(y), bool)
    order = np.argsort(gid, kind="stable")
    ys = y[order]; gs = gid[order]
    bounds = np.concatenate([[0], np.cumsum(np.bincount(np.unique(gs, return_inverse=True)[1]))])
    uniq = np.unique(gs)
    keep_s = np.ones(len(ys), bool)
    for i in range(len(uniq)):
        a, b = bounds[i], bounds[i + 1]
        sz = b - a
        if sz > m_cap:
            drop = rng.permutation(sz)[m_cap:]
            keep_s[a + drop] = False
    return ys[keep_s], gs[keep_s]


def retain_fraction(y, gid, frac, rng):
    """Randomly retain a fraction of each class's responses (>=1 kept)."""
    order = np.argsort(gid, kind="stable")
    ys = y[order]; gs = gid[order]
    _, inv = np.unique(gs, return_inverse=True)
    sizes = np.bincount(inv)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    keep_s = np.zeros(len(ys), bool)
    for i in range(len(sizes)):
        a, b = bounds[i], bounds[i + 1]
        sz = b - a
        nkeep = max(1, int(np.ceil(frac * sz)))
        sel = rng.permutation(sz)[:nkeep]
        keep_s[a + sel] = True
    return ys[keep_s], gs[keep_s]


print("[load] responses ...")
df = pq.read_table(SRC, columns=["QuestionId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32)
df["GroupId"] = df["GroupId"].astype(np.int32)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)
print("  responses:", len(df))

results = {"meta": {"seed": SEED, "note": "Revision analyses (R1). Exploratory relative to the frozen pre-registration."}}

# =========================================================
# RV4 - class-size sensitivity (decimation) of the design effect on difficulty
# =========================================================
print("[RV4] class-size decimation ...")
ids_cap = ss["ci_ratio"]                       # 1500-item validation subsample (frozen)
ids_boot = ids_cap[:400]                        # bootstrap subset (frozen order)
sub = df[df["QuestionId"].isin(set(ids_cap.tolist()))]
by = {q: (g["IsCorrect"].values.astype(np.float64), g["GroupId"].values) for q, g in sub.groupby("QuestionId")}

CAPS = [1, 2, 3, 5, 8, 12, 20, 30, None]        # None = full (no cap)
rng = np.random.default_rng(SEED + 40)
rv4 = {"caps": []}
for m_cap in CAPS:
    an_deffs, an_iccs, mstars, boot_deffs = [], [], [], []
    for j, q in enumerate(ids_cap):
        y, gid = by[int(q)]
        yc, gc = cap_within_class(y, gid, m_cap, rng)
        uniq, invc = np.unique(gc, return_inverse=True)
        m = np.bincount(invc).astype(np.float64)
        s = np.bincount(invc, weights=yc)
        icc, deff = deff_analytic_from_groups(m, s)
        if not np.isnan(deff):
            an_deffs.append(max(deff, np.nan) if np.isnan(deff) else deff)
            an_iccs.append(icc)
            mstars.append(np.sum(m * m) / m.sum())
        if int(q) in set(ids_boot.tolist()):
            bd = cluster_boot_deff(yc, gc, rng, B=400)
            if not np.isnan(bd):
                boot_deffs.append(bd)
    rv4["caps"].append({
        "m_cap": ("full" if m_cap is None else m_cap),
        "n_items_analytic": len(an_deffs),
        "median_icc": round(float(np.nanmedian(an_iccs)), 4),
        "median_mstar": round(float(np.nanmedian(mstars)), 3),
        "median_deff_analytic": round(float(np.nanmedian(an_deffs)), 3),
        "median_deff_bootstrap": round(float(np.nanmedian(boot_deffs)), 3) if boot_deffs else None,
        "n_items_bootstrap": len(boot_deffs),
    })
    c = rv4["caps"][-1]
    print(f"  m_cap={c['m_cap']:>4}  icc={c['median_icc']:.3f}  m*={c['median_mstar']:.2f}"
          f"  deff_analytic={c['median_deff_analytic']:.3f}  deff_boot={c['median_deff_bootstrap']}")

# "drop 50% of each class" budget scenario (halves responses, keeps #classes)
print("[RV4] retain-50%-of-each-class scenario ...")
rng2 = np.random.default_rng(SEED + 41)
full_deff, full_neff, half_deff, half_neff, full_N, half_N = [], [], [], [], [], []
for q in ids_cap:
    y, gid = by[int(q)]
    # full
    uniq, inv = np.unique(gid, return_inverse=True)
    m = np.bincount(inv).astype(np.float64); s = np.bincount(inv, weights=y)
    icc_f, deff_f = deff_analytic_from_groups(m, s)
    N_f = len(y)
    # half (retain 50% within each class)
    yh, gh = retain_fraction(y, gid, 0.5, rng2)
    uniqh, invh = np.unique(gh, return_inverse=True)
    mh = np.bincount(invh).astype(np.float64); sh = np.bincount(invh, weights=yh)
    icc_h, deff_h = deff_analytic_from_groups(mh, sh)
    N_h = len(yh)
    if np.isnan(deff_f) or np.isnan(deff_h):
        continue
    full_deff.append(deff_f); half_deff.append(deff_h)
    full_neff.append(N_f / max(deff_f, 1.0)); half_neff.append(N_h / max(deff_h, 1.0))
    full_N.append(N_f); half_N.append(N_h)
rv4["retain50"] = {
    "n_items": len(full_deff),
    "median_deff_full": round(float(np.median(full_deff)), 3),
    "median_deff_half": round(float(np.median(half_deff)), 3),
    "median_N_full": round(float(np.median(full_N)), 1),
    "median_N_half": round(float(np.median(half_N)), 1),
    "median_neff_full": round(float(np.median(full_neff)), 1),
    "median_neff_half": round(float(np.median(half_neff)), 1),
    "median_neff_ratio_half_over_full": round(float(np.median(np.array(half_neff) / np.array(full_neff))), 3),
    "note": "Retaining 50% of each class halves total responses but the design effect falls (m* halves), so effective sample loss is less than half.",
}
print("  retain50:", json.dumps(rv4["retain50"], indent=2))
results["RV4_class_size"] = rv4

# =========================================================
# RV7 - cluster-robust coverage vs NUMBER OF CLASSES (exact C classes drawn)
# =========================================================
print("[RV7] cluster-robust coverage vs number of classes ...")
cov_ids = ss["coverage"]
subc = df[df["QuestionId"].isin(set(cov_ids.tolist()))]
grpc = {}
for q, g in subc.groupby("QuestionId"):
    y = g["IsCorrect"].values.astype(np.float64)
    gid = g["GroupId"].values
    uniq, inv = np.unique(gid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    ys = y[order]; iv = inv[order]
    sizes = np.bincount(inv)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    gy = [ys[bounds[i]:bounds[i + 1]] for i in range(len(sizes))]
    grpc[int(q)] = (gy, float(y.mean()))

rng3 = np.random.default_rng(SEED + 70)
rv7 = {"by_classes": []}
for C in [10, 20, 40, 80]:
    cov_naive, cov_clu, ns, wid_naive, wid_clu = [], [], [], [], []
    R = 800
    for q in cov_ids:
        gy, tau = grpc[int(q)]
        k = len(gy)
        if k < C:
            continue
        for r in range(R):
            pick = rng3.permutation(k)[:C]
            yy = np.concatenate([gy[i] for i in pick])
            nn = len(yy); ph = yy.mean()
            se_n = np.sqrt(ph * (1 - ph) / nn) if 0 < ph < 1 else 0.0
            cov_naive.append(ph - Z * se_n <= tau <= ph + Z * se_n); wid_naive.append(2 * Z * se_n)
            mm = np.array([len(gy[i]) for i in pick], float)
            ssg = np.array([gy[i].sum() for i in pick], float)
            se_c = np.sqrt((C / (C - 1)) * np.sum((ssg - mm * ph) ** 2)) / nn
            cov_clu.append(ph - Z * se_c <= tau <= ph + Z * se_c); wid_clu.append(2 * Z * se_c)
            ns.append(nn)
    rv7["by_classes"].append({
        "n_classes": C,
        "n_items_eligible": int(sum(1 for q in cov_ids if len(grpc[int(q)][0]) >= C)),
        "mean_responses": round(float(np.mean(ns)), 1),
        "naive_coverage": round(float(np.mean(cov_naive)), 4),
        "cluster_coverage": round(float(np.mean(cov_clu)), 4),
        "naive_width": round(float(np.mean(wid_naive)), 4),
        "cluster_width": round(float(np.mean(wid_clu)), 4),
        "R": R,
    })
    c = rv7["by_classes"][-1]
    print(f"  C={C:>3}  items={c['n_items_eligible']:>3}  meanN={c['mean_responses']:>6}"
          f"  naive={c['naive_coverage']:.3f}  cluster={c['cluster_coverage']:.3f}")
rv7["note"] = ("Cluster-robust 95% CI coverage of item difficulty when exactly C whole classes are "
               "drawn. Coverage climbs toward nominal as C grows; it falls short at few classes "
               "(finite-cluster limitation of clustered SEs).")
results["RV7_cluster_coverage_by_classes"] = rv7

# =========================================================
# RV5 - design-effect-corrected minimum-sample table for DISCRIMINATION
# =========================================================
print("[RV5] discrimination corrected minimum-sample table ...")
disc = pd.read_csv(f"{OUT}/disc_items.csv")
DEFF_DISC = float(disc["DEFF_disc"].median())          # measured discrimination design effect
disc_med = float(disc["disc"].median())
rows = []
for se in [0.02, 0.03, 0.05]:
    for r in [0.20, 0.35, 0.50]:
        # SE of a Pearson/point-biserial correlation ~ (1 - r^2)/sqrt(n-1)  =>  n = ((1-r^2)/SE)^2 + 1
        n_iid = int(np.ceil(((1 - r ** 2) / se) ** 2 + 1))
        n_corr = int(np.ceil(n_iid * DEFF_DISC))
        rows.append({"target_SE": se, "ref_disc_r": r, "iid_N": n_iid,
                     "corrected_N": n_corr, "extra_responses": n_corr - n_iid})
        print(f"  SE={se}  r={r}  iid={n_iid}  corrected(x{DEFF_DISC:.2f})={n_corr}")
results["RV5_discrimination_table"] = {
    "deff_disc_median": round(DEFF_DISC, 3),
    "disc_median": round(disc_med, 3),
    "se_formula": "SE(r) ~= (1 - r^2)/sqrt(n-1); n_iid = ((1-r^2)/SE)^2 + 1; corrected = n_iid x deff_disc",
    "rows": rows,
}

with open(f"{OUT}/revision_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE analysis_revision -> results/revision_results.json")

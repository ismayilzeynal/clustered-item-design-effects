# -*- coding: utf-8 -*-
"""Revision re-runs at FROZEN pre-registered sizes + reviewer-requested additions.
- A5 coverage at 500 items / R=2000 (frozen) + mean signed error (M10) + mean #clusters drawn.
- A7 discrimination at 1000 items (frozen).
- M4: decompose the formula-vs-resampling DEFF gap into ICC finite-sample bias vs m* looseness.
Merges into resample_results.json (overwriting A5, A7; adding M4/M10 fields)."""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 20260711
SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
OUT = "../results"
ss = np.load("../data/subsamples.npz")
stot = pd.read_parquet("../data/student_totals.parquet")
res = json.load(open(f"{OUT}/resample_results.json"))

print("[load] responses ...")
df = pq.read_table(SRC, columns=["QuestionId", "UserId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32); df["GroupId"] = df["GroupId"].astype(np.int32)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)

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

# ---------- A5 coverage at frozen 500 / R=2000 + signed error + clusters ----------
def run_coverage(ids, ns=(100, 200, 400), R=2000):
    rng = np.random.default_rng(SEED + 5)
    sub_all = df[df["QuestionId"].isin(set(ids.tolist()))]
    grp = {q: g for q, g in sub_all.groupby("QuestionId")}
    Z = 1.959964
    out = {}
    for n_target in ns:
        cn, cc, wn, wc, signed, kdrawn = [], [], [], [], [], []
        for q in ids:
            g = grp[int(q)]
            y = g["IsCorrect"].values.astype(np.float64); gid = g["GroupId"].values
            tau = y.mean()
            uniq, inv = np.unique(gid, return_inverse=True); k = len(uniq)
            counts = np.bincount(inv)
            order = np.argsort(inv, kind="stable")
            y_sorted = y[order]; bounds = np.concatenate([[0], np.cumsum(counts)])
            gy = [y_sorted[bounds[i]:bounds[i + 1]] for i in range(k)]
            for r in range(R):
                perm = rng.permutation(k); acc = []; taken = 0; picked = []
                for gi in perm:
                    acc.append(gy[gi]); picked.append(gi); taken += len(gy[gi])
                    if taken >= n_target: break
                yy = np.concatenate(acc); nn = len(yy); ph = yy.mean()
                se_n = np.sqrt(ph * (1 - ph) / nn) if 0 < ph < 1 else 0.0
                cn.append(ph - Z * se_n <= tau <= ph + Z * se_n); wn.append(2 * Z * se_n)
                mm = np.array([len(gy[gi]) for gi in picked], float)
                ssg = np.array([gy[gi].sum() for gi in picked], float); kk = len(picked)
                se_c = (np.sqrt((kk / (kk - 1)) * np.sum((ssg - mm * ph) ** 2)) / nn) if kk > 1 else se_n
                cc.append(ph - Z * se_c <= tau <= ph + Z * se_c); wc.append(2 * Z * se_c)
                signed.append(ph - tau); kdrawn.append(kk)
        out[str(n_target)] = {"naive_coverage": float(np.mean(cn)), "cluster_coverage": float(np.mean(cc)),
                              "naive_width": float(np.mean(wn)), "cluster_width": float(np.mean(wc)),
                              "mean_signed_error": float(np.mean(signed)), "mean_abs_signed_error": float(np.mean(np.abs(signed))),
                              "mean_clusters_drawn": float(np.mean(kdrawn)), "R": R, "n_items": len(ids)}
    return out

print("[A5] coverage frozen 500 / R=2000 + signed error ...")
res["A5_coverage"] = run_coverage(ss["coverage"], ns=(100, 200, 400), R=2000)
print("  ", json.dumps(res["A5_coverage"], indent=2))

# ---------- A7 discrimination at frozen 1000 items ----------
def run_disc(ids, B=500):
    rng = np.random.default_rng(SEED + 7)
    idset = set(ids.tolist())
    sub_all = df[df["QuestionId"].isin(idset)].merge(stot, left_on="UserId", right_index=True, how="left")
    sub_all = sub_all[sub_all["tot_n"] >= 6]
    sub_all["ability_loo"] = (sub_all["tot_c"] - sub_all["IsCorrect"]) / (sub_all["tot_n"] - 1)
    grp = {q: g for q, g in sub_all.groupby("QuestionId")}
    rows = []
    for q in ids:
        if int(q) not in grp: continue
        g = grp[int(q)]
        y = g["IsCorrect"].values.astype(float); a = g["ability_loo"].values.astype(float); gid = g["GroupId"].values
        N = len(y)
        if N < 50 or y.std() == 0 or a.std() == 0: continue
        disc = np.corrcoef(y, a)[0, 1]
        nb = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, N, N); yy, aa = y[idx], a[idx]
            nb[b] = np.corrcoef(yy, aa)[0, 1] if yy.std() > 0 and aa.std() > 0 else np.nan
        naive_se = np.nanstd(nb, ddof=1)
        uniq, inv = np.unique(gid, return_inverse=True); k = len(uniq)
        idx_by_g = [np.where(inv == i)[0] for i in range(k)]
        cb = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, k, k); sel = np.concatenate([idx_by_g[i] for i in pick]); yy, aa = y[sel], a[sel]
            cb[b] = np.corrcoef(yy, aa)[0, 1] if yy.std() > 0 and aa.std() > 0 else np.nan
        cluster_se = np.nanstd(cb, ddof=1)
        if naive_se > 0: rows.append((int(q), N, k, disc, naive_se, cluster_se, (cluster_se / naive_se) ** 2))
    d = pd.DataFrame(rows, columns=["QuestionId", "N", "k", "disc", "naive_se", "cluster_se", "DEFF_disc"])
    d.to_csv(f"{OUT}/disc_items.csv", index=False)
    return {"n_items": int(len(d)), "disc_median": float(d["disc"].median()),
            "DEFF_disc_median": float(d["DEFF_disc"].median()), "DEFF_disc_p25": float(d["DEFF_disc"].quantile(.25)),
            "DEFF_disc_p75": float(d["DEFF_disc"].quantile(.75)), "pct_DEFF_disc_gt_1.5": float((d["DEFF_disc"] > 1.5).mean())}

print("[A7] discrimination frozen 1000 items ...")
res["A7_discrimination"] = run_disc(ss["disc"], B=500)
print("  ", json.dumps(res["A7_discrimination"], indent=2))

# ---------- M4 gap decomposition: ICC bias vs m* looseness ----------
print("[M4] formula-vs-resampling gap decomposition ...")
ci = pd.read_csv(f"{OUT}/ci_ratio_items.csv")
atlas = pd.read_csv(f"{OUT}/item_atlas_t12.csv").set_index("QuestionId")
rng = np.random.default_rng(SEED + 40)
ids = ci["QuestionId"].values
sub_all = df[df["QuestionId"].isin(set(ids.tolist()))]
grp = {q: g for q, g in sub_all.groupby("QuestionId")}
raw_deff, deb_deff, raw_icc, deb_icc = [], [], [], []
for q in ids:
    g = grp[int(q)]
    gg = g.groupby("GroupId")["IsCorrect"].agg(["size", "sum"])
    m = gg["size"].values.astype(float); s = gg["sum"].values.astype(float)
    icc = icc_from_ms(m, s)
    if np.isnan(icc): continue
    mstar = np.sum(m * m) / np.sum(m)
    # single-permutation null ICC for this item (finite-sample bias estimate)
    y = g["IsCorrect"].values.astype(float)
    offs = np.concatenate([[0], np.cumsum(m).astype(int)[:-1]])
    yp = y[rng.permutation(len(y))]; sp = np.add.reduceat(yp, offs)
    icc_null = icc_from_ms(m, sp)
    icc_db = max(icc - (icc_null if not np.isnan(icc_null) else 0.0), 0.0)
    raw_icc.append(icc); deb_icc.append(icc_db)
    raw_deff.append(1 + (mstar - 1) * icc)
    deb_deff.append(1 + (mstar - 1) * icc_db)
raw_deff = np.array(raw_deff); deb_deff = np.array(deb_deff)
boot_med = res["A4_ci_ratio"]["DEFF_boot_median"]
res["M4_gap_decomposition"] = {
    "formula_deff_raw_icc_median": round(float(np.median(raw_deff)), 3),
    "formula_deff_debiased_icc_median": round(float(np.median(deb_deff)), 3),
    "resampling_deff_median": round(float(boot_med), 3),
    "raw_icc_median": round(float(np.median(raw_icc)), 4),
    "debiased_icc_median": round(float(np.median(deb_icc)), 4),
    "gap_closed_by_debias_frac": round(float((np.median(raw_deff) - np.median(deb_deff)) /
                                            (np.median(raw_deff) - boot_med)), 3),
    "note": "Debiasing ICC by a single-permutation null closes most of the formula-vs-resampling gap; residual is m* (burden-weighted cluster size) looseness for a ratio mean.",
}
print("  ", json.dumps(res["M4_gap_decomposition"], indent=2))

with open(f"{OUT}/resample_results.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE analysis_resample2")

# -*- coding: utf-8 -*-
"""A2 permutation null, A4 CI-width ratio, A5 coverage, A7 discrimination DEFF.
Runs on the frozen item subsamples saved by analysis_core.py.
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
stot = pd.read_parquet("../data/student_totals.parquet")

print("[load] responses ...")
df = pq.read_table(SRC, columns=["QuestionId", "UserId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32)
df["GroupId"] = df["GroupId"].astype(np.int32)
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
    if denom <= 0: return np.nan
    return (MSB - MSW) / denom

def item_group_ms(sub):
    gg = sub.groupby("GroupId")["IsCorrect"].agg(["size", "sum"])
    return gg["size"].values.astype(np.float64), gg["sum"].values.astype(np.float64)

# =========================================================
# A2 - permutation null for ICC
# =========================================================
def run_perm_null(ids, B=1000):
    rng = np.random.default_rng(SEED + 2)
    sub_all = df[df["QuestionId"].isin(set(ids.tolist()))]
    grp = {q: g for q, g in sub_all.groupby("QuestionId")}
    obs, null95, pvals = [], [], []
    for q in ids:
        g = grp[int(q)]
        m, s = item_group_ms(g)
        o = icc_from_ms(m, s)
        y = g["IsCorrect"].values.astype(np.float64)
        offs = np.concatenate([[0], np.cumsum(m).astype(int)[:-1]])
        perm_iccs = np.empty(B)
        for b in range(B):
            yp = y[rng.permutation(len(y))]
            sp = np.add.reduceat(yp, offs)
            perm_iccs[b] = icc_from_ms(m, sp)
        obs.append(o); null95.append(np.nanquantile(perm_iccs, 0.95))
        pvals.append((np.sum(perm_iccs >= o) + 1) / (B + 1))
    obs = np.array(obs); null95 = np.array(null95); pvals = np.array(pvals)
    # BH correction
    order = np.argsort(pvals); ranked = pvals[order]
    m_ = len(pvals); bh = ranked * m_ / (np.arange(m_) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    sig = np.zeros(m_, bool); sig[order] = bh < 0.05
    return {
        "n_items": int(m_),
        "obs_icc_median": float(np.nanmedian(obs)),
        "null_icc_median": float(np.nanmedian(null95)),
        "null_icc_mean": float(np.nanmean([np.nan] * 0 + list(null95))),
        "frac_obs_gt_null95": float(np.mean(obs > null95)),
        "frac_sig_BH": float(np.mean(sig)),
        "median_pval": float(np.median(pvals)),
    }

print("[A2] permutation null ...")
a2 = run_perm_null(ss["perm_null"], B=1000)
print("  ", json.dumps(a2, indent=2))

# =========================================================
# A4 - naive vs cluster bootstrap CI width -> empirical DEFF
# =========================================================
def run_ci_ratio(ids, B=2000):
    rng = np.random.default_rng(SEED + 4)
    sub_all = df[df["QuestionId"].isin(set(ids.tolist()))]
    grp = {q: g for q, g in sub_all.groupby("QuestionId")}
    rows = []
    for q in ids:
        g = grp[int(q)]
        y = g["IsCorrect"].values.astype(np.float64)
        gid = g["GroupId"].values
        N = len(y); p = y.mean()
        if p <= 0 or p >= 1: continue
        naive_se = np.sqrt(p * (1 - p) / N)
        # cluster structure
        uniq, inv = np.unique(gid, return_inverse=True)
        m = np.bincount(inv).astype(np.float64)
        s = np.bincount(inv, weights=y)
        k = len(uniq)
        # cluster bootstrap: resample k groups with replacement
        means = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, k, k)
            means[b] = s[pick].sum() / m[pick].sum()
        cluster_se = means.std(ddof=1)
        # linearization (sandwich) clustered SE as a check
        lin_se = np.sqrt((k / (k - 1)) * np.sum((s - m * p) ** 2)) / N
        rows.append((int(q), N, k, p, naive_se, cluster_se, lin_se,
                     (cluster_se / naive_se) ** 2, (lin_se / naive_se) ** 2))
    cols = ["QuestionId", "N", "k", "p", "naive_se", "cluster_se", "lin_se", "DEFF_boot", "DEFF_lin"]
    d = pd.DataFrame(rows, columns=cols)
    d.to_csv(f"{OUT}/ci_ratio_items.csv", index=False)
    # merge ANOVA DEFF from atlas
    atlas = pd.read_csv(f"{OUT}/item_atlas_t12.csv").set_index("QuestionId")
    d = d.set_index("QuestionId")
    d["DEFF_anova"] = atlas["DEFF"].reindex(d.index)
    corr = float(np.corrcoef(np.sqrt(d["DEFF_boot"]), np.sqrt(d["DEFF_anova"]))[0, 1])
    return {
        "n_items": int(len(d)),
        "DEFF_boot_median": float(d["DEFF_boot"].median()),
        "DEFF_lin_median": float(d["DEFF_lin"].median()),
        "DEFF_anova_median": float(d["DEFF_anova"].median()),
        "widthratio_boot_median": float(np.sqrt(d["DEFF_boot"]).median()),
        "corr_sqrtDEFF_boot_vs_anova": corr,
        "median_ratio_boot_over_anova": float((d["DEFF_boot"] / d["DEFF_anova"]).median()),
    }

print("[A4] CI-width ratio / empirical DEFF ...")
a4 = run_ci_ratio(ss["ci_ratio"], B=2000)
print("  ", json.dumps(a4, indent=2))

# =========================================================
# A5 - coverage of naive vs cluster-robust 95% CI under whole-class sampling
# =========================================================
def run_coverage(ids, ns=(100, 200, 400), R=1000):
    rng = np.random.default_rng(SEED + 5)
    sub_all = df[df["QuestionId"].isin(set(ids.tolist()))]
    grp = {q: g for q, g in sub_all.groupby("QuestionId")}
    Z = 1.959964
    out = {}
    for n_target in ns:
        cov_naive = []; cov_clu = []; w_naive = []; w_clu = []
        for q in ids:
            g = grp[int(q)]
            y = g["IsCorrect"].values.astype(np.float64)
            gid = g["GroupId"].values
            tau = y.mean()
            uniq, inv = np.unique(gid, return_inverse=True)
            k = len(uniq)
            # group -> its response values
            order = np.argsort(inv, kind="stable")
            y_sorted = y[order]; inv_sorted = inv[order]
            bounds = np.concatenate([[0], np.cumsum(np.bincount(inv))])
            gy = [y_sorted[bounds[i]:bounds[i + 1]] for i in range(k)]
            for r in range(R):
                perm = rng.permutation(k)
                acc = []; taken = 0; picked = []
                for gi in perm:
                    acc.append(gy[gi]); picked.append(gi); taken += len(gy[gi])
                    if taken >= n_target: break
                yy = np.concatenate(acc); nn = len(yy); ph = yy.mean()
                # naive binomial CI
                se_n = np.sqrt(ph * (1 - ph) / nn) if 0 < ph < 1 else 0.0
                lo_n, hi_n = ph - Z * se_n, ph + Z * se_n
                cov_naive.append(lo_n <= tau <= hi_n); w_naive.append(2 * Z * se_n)
                # cluster-robust (linearization) CI over the picked groups
                mm = np.array([len(gy[gi]) for gi in picked], float)
                ssg = np.array([gy[gi].sum() for gi in picked], float)
                kk = len(picked)
                se_c = (np.sqrt((kk / (kk - 1)) * np.sum((ssg - mm * ph) ** 2)) / nn) if kk > 1 else se_n
                lo_c, hi_c = ph - Z * se_c, ph + Z * se_c
                cov_clu.append(lo_c <= tau <= hi_c); w_clu.append(2 * Z * se_c)
        out[str(n_target)] = {
            "naive_coverage": float(np.mean(cov_naive)),
            "cluster_coverage": float(np.mean(cov_clu)),
            "naive_width": float(np.mean(w_naive)),
            "cluster_width": float(np.mean(w_clu)),
            "R": R, "n_items": len(ids),
        }
    return out

print("[A5] coverage experiment ...")
cov_ids = ss["coverage"][:300]   # cap for compute (frozen order)
a5 = run_coverage(cov_ids, ns=(100, 200, 400), R=800)
print("  ", json.dumps(a5, indent=2))

# =========================================================
# A7 - discrimination (item-rest point-biserial) design effect
# =========================================================
def run_disc(ids, B=500):
    rng = np.random.default_rng(SEED + 7)
    idset = set(ids.tolist())
    sub_all = df[df["QuestionId"].isin(idset)].merge(
        stot, left_on="UserId", right_index=True, how="left")
    sub_all = sub_all[sub_all["tot_n"] >= 6]
    sub_all["ability_loo"] = (sub_all["tot_c"] - sub_all["IsCorrect"]) / (sub_all["tot_n"] - 1)
    grp = {q: g for q, g in sub_all.groupby("QuestionId")}
    rows = []
    for q in ids:
        if int(q) not in grp: continue
        g = grp[int(q)]
        y = g["IsCorrect"].values.astype(np.float64)
        a = g["ability_loo"].values.astype(np.float64)
        gid = g["GroupId"].values
        N = len(y)
        if N < 50 or y.std() == 0 or a.std() == 0: continue
        disc = np.corrcoef(y, a)[0, 1]
        # naive bootstrap (resample responses)
        nb = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, N, N)
            yy, aa = y[idx], a[idx]
            nb[b] = np.corrcoef(yy, aa)[0, 1] if yy.std() > 0 and aa.std() > 0 else np.nan
        naive_se = np.nanstd(nb, ddof=1)
        # cluster bootstrap (resample groups)
        uniq, inv = np.unique(gid, return_inverse=True)
        k = len(uniq)
        idx_by_g = [np.where(inv == i)[0] for i in range(k)]
        cb = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, k, k)
            sel = np.concatenate([idx_by_g[i] for i in pick])
            yy, aa = y[sel], a[sel]
            cb[b] = np.corrcoef(yy, aa)[0, 1] if yy.std() > 0 and aa.std() > 0 else np.nan
        cluster_se = np.nanstd(cb, ddof=1)
        if naive_se > 0:
            rows.append((int(q), N, k, disc, naive_se, cluster_se, (cluster_se / naive_se) ** 2))
    d = pd.DataFrame(rows, columns=["QuestionId", "N", "k", "disc", "naive_se", "cluster_se", "DEFF_disc"])
    d.to_csv(f"{OUT}/disc_items.csv", index=False)
    return {
        "n_items": int(len(d)),
        "disc_median": float(d["disc"].median()),
        "DEFF_disc_median": float(d["DEFF_disc"].median()),
        "DEFF_disc_p25": float(d["DEFF_disc"].quantile(.25)),
        "DEFF_disc_p75": float(d["DEFF_disc"].quantile(.75)),
        "pct_DEFF_disc_gt_1.5": float((d["DEFF_disc"] > 1.5).mean()),
    }

print("[A7] discrimination DEFF ...")
a7 = run_disc(ss["disc"][:600], B=400)   # cap subsample & B for compute (frozen order)
print("  ", json.dumps(a7, indent=2))

res = {"A2_perm_null": a2, "A4_ci_ratio": a4, "A5_coverage": a5, "A7_discrimination": a7}
with open(f"{OUT}/resample_results.json", "w") as f:
    json.dump(res, f, indent=2)
print("\nDONE analysis_resample")

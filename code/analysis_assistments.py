# -*- coding: utf-8 -*-
"""Second corpus (T3 generalization): ASSISTments 2009 skill-builder.
Structurally different platform (US intelligent tutoring; class/teacher/school clustering).
Replicates the classroom-ICC / design-effect atlas on item (problem) difficulty.
"""
import json, os
import numpy as np
import pandas as pd

SEED = 20260711
rng = np.random.default_rng(SEED + 20)
SRC = os.environ.get("ASSIST09", "../data/assistments09_skill_builder.csv")
OUT = "../results"

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

def atlas(df, item_col, clus_col, nmin, kmin):
    ig = df.groupby([item_col, clus_col], sort=False)["correct"].agg(["size", "sum"]).reset_index()
    ig.columns = ["item", "clus", "m", "s"]
    ig["m"] = ig["m"].astype(float); ig["s"] = ig["s"].astype(float)
    ig["s2_over_m"] = ig["s"] ** 2 / ig["m"]; ig["m2"] = ig["m"] ** 2
    g = ig.groupby("item")
    per = pd.DataFrame({"N": g["m"].sum(), "S": g["s"].sum(), "k": g.size(),
                        "sum_s2_over_m": g["s2_over_m"].sum(), "sum_m2": g["m2"].sum()})
    per["p"] = per["S"] / per["N"]
    per["SSB"] = per["sum_s2_over_m"] - per["S"] ** 2 / per["N"]
    per["SST"] = per["S"] * (per["N"] - per["S"]) / per["N"]
    per["SSW"] = per["SST"] - per["SSB"]
    per["MSB"] = per["SSB"] / (per["k"] - 1); per["MSW"] = per["SSW"] / (per["N"] - per["k"])
    per["m0"] = (per["N"] - per["sum_m2"] / per["N"]) / (per["k"] - 1)
    per["ICC"] = (per["MSB"] - per["MSW"]) / (per["MSB"] + (per["m0"] - 1) * per["MSW"])
    per["mstar"] = per["sum_m2"] / per["N"]
    per["DEFF"] = 1 + (per["mstar"] - 1) * per["ICC"]
    sel = per[(per["N"] >= nmin) & (per["k"] >= kmin) & (per["p"] >= 0.02) & (per["p"] <= 0.98)].copy()
    return sel, ig

def cluster_boot_deff(df, ids, item_col, clus_col, B=2000):
    sub = df[df[item_col].isin(set(ids))]
    grp = {q: g for q, g in sub.groupby(item_col)}
    out = []
    for q in ids:
        g = grp[q]
        y = g["correct"].values.astype(float); gid = g[clus_col].values
        N = len(y); p = y.mean()
        if p <= 0 or p >= 1: continue
        naive_se = np.sqrt(p * (1 - p) / N)
        uniq, inv = np.unique(gid, return_inverse=True)
        m = np.bincount(inv).astype(float); s = np.bincount(inv, weights=y); k = len(uniq)
        means = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, k, k)
            means[b] = s[pick].sum() / m[pick].sum()
        out.append((means.std(ddof=1) / naive_se) ** 2)
    return np.array(out)

print("[load] ASSISTments 2009 ...")
df = pd.read_csv(SRC, encoding="latin-1",
                 usecols=["user_id", "problem_id", "correct", "student_class_id", "template_id", "school_id"],
                 low_memory=False)
df = df.dropna(subset=["problem_id", "student_class_id", "correct"])
df = df[df["correct"].isin([0, 1])]
df["correct"] = df["correct"].astype(np.int8)
print(f"  {len(df):,} responses | {df.problem_id.nunique():,} problems | {df.student_class_id.nunique()} classes | {df.school_id.nunique()} schools | {df.user_id.nunique():,} students")

res = {"dataset": "ASSISTments 2009 skill-builder", "n_responses": int(len(df)),
       "n_problems": int(df.problem_id.nunique()), "n_classes": int(df.student_class_id.nunique()),
       "n_schools": int(df.school_id.nunique()), "n_students": int(df.user_id.nunique()),
       "overall_correct_rate": round(float(df.correct.mean()), 4)}

# primary: problem-level, class cluster, >=100 resp & >=10 classes
sel, _ = atlas(df, "problem_id", "student_class_id", nmin=100, kmin=10)
db = cluster_boot_deff(df, list(sel.index), "problem_id", "student_class_id", B=2000)
res["problem_class"] = {
    "inclusion": ">=100 resp & >=10 classes", "n_items": int(len(sel)),
    "median_resp": float(sel["N"].median()), "median_classes": float(sel["k"].median()),
    "icc_median": round(float(sel["ICC"].median()), 4), "icc_p25": round(float(sel["ICC"].quantile(.25)), 4),
    "icc_p75": round(float(sel["ICC"].quantile(.75)), 4), "pct_icc_gt_0.05": round(float((sel["ICC"] > 0.05).mean()), 4),
    "deff_formula_median": round(float(sel["DEFF"].median()), 3),
    "deff_resampling_median": round(float(np.median(db)), 3),
    "pct_deff_gt_1.5": round(float((sel["DEFF"] > 1.5).mean()), 4),
}
sel.reset_index().rename(columns={"item": "problem_id"})[["problem_id", "N", "k", "p", "ICC", "mstar", "DEFF"]].round(6).to_csv(f"{OUT}/assistments_problem_atlas.csv", index=False)

# robustness: template-level (item families)
selt, _ = atlas(df, "template_id", "student_class_id", nmin=200, kmin=20)
res["template_class"] = {"inclusion": ">=200 resp & >=20 classes", "n_items": int(len(selt)),
                         "icc_median": round(float(selt["ICC"].median()), 4),
                         "deff_formula_median": round(float(selt["DEFF"].median()), 3)}

# robustness: school as coarser cluster (problem-level)
sels, _ = atlas(df, "problem_id", "school_id", nmin=100, kmin=5)
res["problem_school"] = {"inclusion": ">=100 resp & >=5 schools", "n_items": int(len(sels)),
                         "icc_median": round(float(sels["ICC"].median()), 4),
                         "deff_formula_median": round(float(sels["DEFF"].median()), 3)}

with open(f"{OUT}/assistments_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
print("DONE analysis_assistments")

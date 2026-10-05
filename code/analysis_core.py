# -*- coding: utf-8 -*-
"""A1 + A3 + A8 : per-item ANOVA ICC atlas, design effects, heterogeneity, H1/H2.

Vectorized: per-item ANOVA ICC is computed from (item, group) aggregates
[m_ig, s_ig], where s_ig = sum of IsCorrect. All items in the primary set.
Writes per-item atlas CSV, subject/difficulty breakdowns, student totals (for A7),
fixed random item subsamples (for A2/A4/A5/A7), and headline JSON.
"""
import json, os, zipfile, io, hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 20260711
rng = np.random.default_rng(SEED)
SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
ZIP = os.environ.get("EEDI_ZIP", "../data/eedi_challenge.zip")
OUT = "../results"
import os; os.makedirs(OUT, exist_ok=True); os.makedirs("../data", exist_ok=True)

MIN_RESP, MIN_GROUPS = 200, 20   # primary inclusion (frozen)

print("[load] responses ...")
df = pq.read_table(SRC, columns=["QuestionId", "UserId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32)
df["GroupId"] = df["GroupId"].astype(np.int32)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)
print("  responses:", len(df))

# ---- student totals (for A7 leave-item-out ability), saved ----
stot = df.groupby("UserId")["IsCorrect"].agg(["sum", "size"]).rename(columns={"sum": "tot_c", "size": "tot_n"})
stot.to_parquet("../data/student_totals.parquet")
print("  saved student totals:", len(stot))

# ---- (item, group) aggregates ----
print("[agg] item-group cells ...")
ig = df.groupby(["QuestionId", "GroupId"], sort=False)["IsCorrect"].agg(["size", "sum"]).reset_index()
ig.columns = ["QuestionId", "GroupId", "m", "s"]
ig["m"] = ig["m"].astype(np.int64); ig["s"] = ig["s"].astype(np.int64)
ig["s2_over_m"] = ig["s"] * ig["s"] / ig["m"]
ig["m2"] = ig["m"] * ig["m"]

# ---- per-item sums ----
g = ig.groupby("QuestionId")
per = pd.DataFrame({
    "N": g["m"].sum(),
    "S": g["s"].sum(),
    "k": g.size(),
    "sum_s2_over_m": g["s2_over_m"].sum(),
    "sum_m2": g["m2"].sum(),
})
per["p"] = per["S"] / per["N"]
# ANOVA sums of squares
per["SSB"] = per["sum_s2_over_m"] - per["S"] * per["S"] / per["N"]
per["SST"] = per["S"] * (per["N"] - per["S"]) / per["N"]
per["SSW"] = per["SST"] - per["SSB"]
per["MSB"] = per["SSB"] / (per["k"] - 1)
per["MSW"] = per["SSW"] / (per["N"] - per["k"])
per["m0"] = (per["N"] - per["sum_m2"] / per["N"]) / (per["k"] - 1)          # ANOVA design cluster size
per["ICC"] = (per["MSB"] - per["MSW"]) / (per["MSB"] + (per["m0"] - 1) * per["MSW"])
per["mstar"] = per["sum_m2"] / per["N"]                                     # Kish burden-weighted mean cluster size
per["DEFF"] = 1 + (per["mstar"] - 1) * per["ICC"]
per["n_eff"] = per["N"] / per["DEFF"]

# primary inclusion
prim = per[(per["N"] >= MIN_RESP) & (per["k"] >= MIN_GROUPS) & (per["p"] >= 0.02) & (per["p"] <= 0.98)].copy()
# guard: ICC can be slightly negative under null; clip DEFF at 1 for the corrected-N use but keep raw ICC for atlas
prim["ICC_raw"] = prim["ICC"]
prim["DEFF_use"] = np.maximum(prim["DEFF"], 1.0)
print(f"[primary] {len(prim)} items (>= {MIN_RESP} resp & >= {MIN_GROUPS} groups)")

def q(s, xs=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    return {f"p{int(x*100)}": float(np.quantile(s, x)) for x in xs}

def item_boot_median(vals, B=2000):
    vals = np.asarray(vals, float)
    n = len(vals)
    meds = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        meds[b] = np.median(vals[idx])
    return float(np.median(vals)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))

icc_med, icc_lo, icc_hi = item_boot_median(prim["ICC_raw"].values)
deff_med, deff_lo, deff_hi = item_boot_median(prim["DEFF"].values)
print(f"[H1] median ICC = {icc_med:.4f}  95% CI [{icc_lo:.4f}, {icc_hi:.4f}]  (> 0.05 ? {icc_lo>0.05})")
print(f"[H2] median DEFF = {deff_med:.3f}  95% CI [{deff_lo:.3f}, {deff_hi:.3f}]  (> 1.5 ? {deff_lo>1.5})")

# ---- subject mapping (top-level subject) ----
z = zipfile.ZipFile(ZIP)
with z.open("data/metadata/question_metadata_task_1_2.csv") as f:
    qm = pd.read_csv(f)
with z.open("data/metadata/subject_metadata.csv") as f:
    subj = pd.read_csv(f)
# subject_metadata: SubjectId, Name, ParentId, Level
lvl1 = subj[subj["Level"] == 1].set_index("SubjectId")["Name"].to_dict()
# parent chain to level 1
parent = subj.set_index("SubjectId")["ParentId"].to_dict()
level = subj.set_index("SubjectId")["Level"].to_dict()
name = subj.set_index("SubjectId")["Name"].to_dict()
def to_level1(sid):
    cur = sid
    seen = 0
    while cur in parent and level.get(cur, 99) > 1 and seen < 20:
        nxt = parent[cur]
        if pd.isna(nxt): break
        cur = int(nxt); seen += 1
    return name.get(cur, "Unknown")
import ast
qm["subjects"] = qm["SubjectId"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
# choose the most specific-but-mapped-to-level1: use the last subject id's level-1 ancestor
qm["subj1"] = qm["subjects"].apply(lambda lst: to_level1(int(lst[-1])) if lst else "Unknown")
q2subj = qm.set_index("QuestionId")["subj1"].to_dict()
prim["subject"] = prim.index.map(q2subj).fillna("Unknown")

# difficulty band
def band(p):
    if p < 0.4: return "hard(.02-.4)"
    if p < 0.6: return "medium(.4-.6)"
    if p < 0.8: return "easy(.6-.8)"
    return "very_easy(.8-.98)"
prim["diff_band"] = prim["p"].apply(band)

subj_break = prim.groupby("subject").agg(n_items=("ICC_raw", "size"), icc_med=("ICC_raw", "median"),
                                         deff_med=("DEFF", "median"), mstar_med=("mstar", "median")).sort_values("n_items", ascending=False)
band_break = prim.groupby("diff_band").agg(n_items=("ICC_raw", "size"), icc_med=("ICC_raw", "median"),
                                           deff_med=("DEFF", "median")).reindex(["hard(.02-.4)","medium(.4-.6)","easy(.6-.8)","very_easy(.8-.98)"])
print("\n[A8 subject]\n", subj_break.round(3).to_string())
print("\n[A8 difficulty band]\n", band_break.round(3).to_string())

# ---- save atlas (DERIVED aggregates only, redistributable) ----
atlas = prim[["N", "k", "p", "ICC_raw", "mstar", "m0", "DEFF", "DEFF_use", "n_eff", "subject", "diff_band"]].copy()
atlas.index.name = "QuestionId"
atlas.round(6).to_csv(f"{OUT}/item_atlas_t12.csv")
subj_break.round(6).to_csv(f"{OUT}/subject_breakdown.csv")
band_break.round(6).to_csv(f"{OUT}/difficulty_breakdown.csv")

# ---- fixed random item subsamples for resample analyses (frozen) ----
prim_ids = prim.index.values.copy()
rng2 = np.random.default_rng(SEED + 1)
subsamples = {
    "perm_null": rng2.choice(prim_ids, size=min(1000, len(prim_ids)), replace=False),
    "ci_ratio": rng2.choice(prim_ids, size=min(1500, len(prim_ids)), replace=False),
    "coverage": rng2.choice(prim[prim["N"] >= 1500].index.values, size=min(500, int((prim["N"]>=1500).sum())), replace=False),
    "disc": rng2.choice(prim_ids, size=min(1000, len(prim_ids)), replace=False),
}
np.savez("../data/subsamples.npz", **{k: v for k, v in subsamples.items()})
print("\n[subsamples] sizes:", {k: len(v) for k, v in subsamples.items()})

# ---- headline JSON ----
head = {
    "seed": SEED, "min_resp": MIN_RESP, "min_groups": MIN_GROUPS,
    "n_responses": int(len(df)), "n_items_total": int(per.shape[0]), "n_items_primary": int(len(prim)),
    "n_groups": int(df["GroupId"].nunique()), "n_students": int(df["UserId"].nunique()),
    "overall_correct_rate": float(df["IsCorrect"].mean()),
    "median_responses_per_item": float(prim["N"].median()),
    "median_groups_per_item": float(prim["k"].median()),
    "median_mstar": float(prim["mstar"].median()),
    "H1_median_ICC": icc_med, "H1_ci": [icc_lo, icc_hi], "H1_supported": bool(icc_lo > 0.05),
    "H2_median_DEFF": deff_med, "H2_ci": [deff_lo, deff_hi], "H2_supported": bool(deff_lo > 1.5),
    "ICC_quantiles": q(prim["ICC_raw"]), "DEFF_quantiles": q(prim["DEFF"]),
    "n_eff_ratio_median": float((prim["n_eff"] / prim["N"]).median()),
    "pct_items_DEFF_gt_1.5": float((prim["DEFF"] > 1.5).mean()),
    "pct_items_DEFF_gt_2": float((prim["DEFF"] > 2).mean()),
    "pct_items_ICC_gt_0.05": float((prim["ICC_raw"] > 0.05).mean()),
}
with open(f"{OUT}/core_headline.json", "w") as f:
    json.dump(head, f, indent=2)
print("\n[headline]\n", json.dumps(head, indent=2))
print("\nDONE analysis_core")

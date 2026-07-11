# -*- coding: utf-8 -*-
"""A9 robustness: (a) inclusion thresholds, (b) latent logistic ICC, (c) QuizId cluster,
   (d) cap max cluster size, (e) min group size>=2, (f) Task 3&4 replication."""
import json, os, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")

SEED = 20260711
rng = np.random.default_rng(SEED + 9)
T12 = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
T34 = os.environ.get("EEDI_T34", "../data/resp_t34.parquet")
OUT = "../results"

def per_item_from_cells(cells):
    """cells: DataFrame with columns [item, m, s]; returns per-item ICC/DEFF frame."""
    cells = cells.copy()
    cells["s2_over_m"] = cells["s"] * cells["s"] / cells["m"]
    cells["m2"] = cells["m"] * cells["m"]
    g = cells.groupby("item")
    per = pd.DataFrame({"N": g["m"].sum(), "S": g["s"].sum(), "k": g.size(),
                        "sum_s2_over_m": g["s2_over_m"].sum(), "sum_m2": g["m2"].sum()})
    per["p"] = per["S"] / per["N"]
    per["SSB"] = per["sum_s2_over_m"] - per["S"] ** 2 / per["N"]
    per["SST"] = per["S"] * (per["N"] - per["S"]) / per["N"]
    per["SSW"] = per["SST"] - per["SSB"]
    per["MSB"] = per["SSB"] / (per["k"] - 1)
    per["MSW"] = per["SSW"] / (per["N"] - per["k"])
    per["m0"] = (per["N"] - per["sum_m2"] / per["N"]) / (per["k"] - 1)
    per["ICC"] = (per["MSB"] - per["MSW"]) / (per["MSB"] + (per["m0"] - 1) * per["MSW"])
    per["mstar"] = per["sum_m2"] / per["N"]
    per["DEFF"] = 1 + (per["mstar"] - 1) * per["ICC"]
    return per

def cells_from_df(df, item_col, clus_col):
    ig = df.groupby([item_col, clus_col], sort=False)["IsCorrect"].agg(["size", "sum"]).reset_index()
    ig.columns = ["item", "clus", "m", "s"]
    return ig[["item", "m", "s"]].assign(m=ig["m"].astype(float), s=ig["s"].astype(float)), ig

print("[load] T12 ...")
df = pq.read_table(T12, columns=["QuestionId", "IsCorrect", "GroupId", "QuizId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect", "QuizId"])
for c in ["QuestionId", "GroupId", "QuizId"]:
    df[c] = df[c].astype(np.int64)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)

out = {}

# ---- (a) inclusion thresholds (GroupId cluster) ----
cells_g, ig_g = cells_from_df(df, "QuestionId", "GroupId")
per_full = per_item_from_cells(cells_g)
thr = {}
for nmin in [100, 200, 500]:
    for kmin in [10, 20, 50]:
        sel = per_full[(per_full["N"] >= nmin) & (per_full["k"] >= kmin) &
                       (per_full["p"] >= 0.02) & (per_full["p"] <= 0.98)]
        thr[f"N>={nmin},k>={kmin}"] = {"n_items": int(len(sel)),
                                       "icc_med": float(sel["ICC"].median()),
                                       "deff_med": float(sel["DEFF"].median())}
out["a_thresholds"] = thr
print("[A9a] thresholds:", json.dumps(thr, indent=2))

# ---- primary set indices for the other checks ----
prim = per_full[(per_full["N"] >= 200) & (per_full["k"] >= 20) & (per_full["p"] >= 0.02) & (per_full["p"] <= 0.98)]
prim_ids = set(prim.index.tolist())

# ---- (c) QuizId as cluster ----
dfq = df[df["QuestionId"].isin(prim_ids)]
cells_q, _ = cells_from_df(dfq, "QuestionId", "QuizId")
per_q = per_item_from_cells(cells_q)
per_q = per_q[(per_q["k"] >= 20)]
out["c_quizid_cluster"] = {"n_items": int(len(per_q)), "icc_med": float(per_q["ICC"].median()),
                           "deff_med": float(per_q["DEFF"].median())}
# quiz-template spread across classes (why QuizId is a coarser, crossed unit)
gpq = df.groupby("QuizId")["GroupId"].nunique()
out["c_quizid_spread"] = {"n_quizzes": int(df["QuizId"].nunique()),
                          "groups_per_quiz_median": int(gpq.median()),
                          "groups_per_quiz_mean": round(float(gpq.mean()), 2),
                          "groups_per_quiz_max": int(gpq.max())}
print("[A9c] QuizId cluster:", out["c_quizid_cluster"], "| spread:", out["c_quizid_spread"])

# ---- (d) cap max cluster size at 30 (subsample larger groups) ----
def cap_cells(ig, cap=30):
    ig = ig.copy()
    over = ig["m"] > cap
    # scale correct count proportionally when capping (keeps p, reduces m)
    ig.loc[over, "s"] = np.round(ig.loc[over, "s"] * cap / ig.loc[over, "m"])
    ig.loc[over, "m"] = cap
    return ig.rename(columns={"clus": "drop"})[["item", "m", "s"]].astype({"m": float, "s": float})
ig_g2 = ig_g.copy(); ig_g2.columns = ["item", "clus", "m", "s"]
capped = cap_cells(ig_g2[ig_g2["item"].isin(prim_ids)], cap=30)
per_cap = per_item_from_cells(capped)
per_cap = per_cap[per_cap["k"] >= 20]
out["d_cap30"] = {"n_items": int(len(per_cap)), "icc_med": float(per_cap["ICC"].median()),
                  "deff_med": float(per_cap["DEFF"].median())}
print("[A9d] cap m<=30:", out["d_cap30"])

# ---- (e) drop singleton groups (m>=2) ----
ig_ns = ig_g2[(ig_g2["item"].isin(prim_ids)) & (ig_g2["m"] >= 2)][["item", "m", "s"]].astype({"m": float, "s": float})
per_ns = per_item_from_cells(ig_ns)
per_ns = per_ns[per_ns["k"] >= 20]
out["e_min_group2"] = {"n_items": int(len(per_ns)), "icc_med": float(per_ns["ICC"].median()),
                       "deff_med": float(per_ns["DEFF"].median())}
print("[A9e] m>=2:", out["e_min_group2"])

# ---- (f) Task 3&4 replication ----
print("[load] T34 ...")
d34 = pq.read_table(T34, columns=["QuestionId", "IsCorrect", "GroupId"]).to_pandas()
d34 = d34.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
d34["QuestionId"] = d34["QuestionId"].astype(np.int64); d34["GroupId"] = d34["GroupId"].astype(np.int64)
d34["IsCorrect"] = d34["IsCorrect"].astype(np.int8)
cells34, _ = cells_from_df(d34, "QuestionId", "GroupId")
per34 = per_item_from_cells(cells34)
sel34 = per34[(per34["N"] >= 200) & (per34["k"] >= 20) & (per34["p"] >= 0.02) & (per34["p"] <= 0.98)]
out["f_task34"] = {"n_responses": int(len(d34)), "n_items": int(len(sel34)),
                   "icc_med": float(sel34["ICC"].median()), "deff_med": float(sel34["DEFF"].median())}
print("[A9f] Task3&4 replication:", out["f_task34"])

# ---- (b) latent logistic (random-intercept) ICC on a subsample ----
latent = {"status": "skipped"}
try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    ids = rng.choice(list(prim_ids), size=120, replace=False)
    sub = df[df["QuestionId"].isin(set(ids.tolist()))]
    lat_icc, lin_icc = [], []
    for q in ids:
        g = sub[sub["QuestionId"] == q][["IsCorrect", "GroupId"]].copy()
        if g["GroupId"].nunique() < 10: continue
        try:
            m = BinomialBayesMixedGLM.from_formula("IsCorrect ~ 1", {"grp": "0 + C(GroupId)"}, g)
            r = m.fit_vb()
            sd = float(np.exp(r.vcp_mean[0]))   # random-effect SD (log-scale param)
            icc_l = sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)
            lat_icc.append(icc_l)
            lin_icc.append(float(per_full.loc[q, "ICC"]))
        except Exception:
            continue
    if lat_icc:
        latent = {"status": "ok", "n_items": len(lat_icc),
                  "latent_icc_med": float(np.median(lat_icc)),
                  "linear_icc_med_same_items": float(np.median(lin_icc))}
except Exception as e:
    latent = {"status": "error", "msg": str(e)[:200]}
out["b_latent_logistic_icc"] = latent
print("[A9b] latent logistic ICC:", latent)

with open(f"{OUT}/robust_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE analysis_robust")

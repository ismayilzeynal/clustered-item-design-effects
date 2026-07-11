# -*- coding: utf-8 -*-
"""M9: classroom variation NET OF quiz template (group-within-quiz ICC).
A9b (frozen): latent random-intercept logistic ICC on 300 items.
Merges into robust_results.json."""
import json, os, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")

SEED = 20260711
rng = np.random.default_rng(SEED + 90)
T12 = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
OUT = "../results"
rob = json.load(open(f"{OUT}/robust_results.json"))
atlas = pd.read_csv(f"{OUT}/item_atlas_t12.csv").set_index("QuestionId")
prim_ids = atlas.index.values

print("[load] T12 with QuizId ...")
df = pq.read_table(T12, columns=["QuestionId", "IsCorrect", "GroupId", "QuizId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect", "QuizId"])
for c in ["QuestionId", "GroupId", "QuizId"]:
    df[c] = df[c].astype(np.int64)
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

# ---------- M9: between-group ICC WITHIN the same quiz ----------
# For each (item, quiz) stratum with >=3 groups and >=30 responses, compute the ANOVA ICC
# of correctness across GroupId. If classrooms differ even on the SAME quiz template, the
# classroom effect is net of the quiz -> classroom is the binding unit.
print("[M9] group-within-quiz ICC ...")
ids = rng.choice(prim_ids, size=1500, replace=False)
sub = df[df["QuestionId"].isin(set(ids.tolist()))]
within_iccs = []
for (q, quiz), g in sub.groupby(["QuestionId", "QuizId"]):
    gg = g.groupby("GroupId")["IsCorrect"].agg(["size", "sum"])
    if len(gg) < 3 or gg["size"].sum() < 30:
        continue
    icc = icc_from_ms(gg["size"].values.astype(float), gg["sum"].values.astype(float))
    if not np.isnan(icc):
        within_iccs.append(icc)
within_iccs = np.array(within_iccs)
rob["m9_group_within_quiz"] = {
    "n_strata": int(len(within_iccs)),
    "within_quiz_between_group_icc_median": round(float(np.median(within_iccs)), 4),
    "within_quiz_between_group_icc_p25": round(float(np.quantile(within_iccs, .25)), 4),
    "within_quiz_between_group_icc_p75": round(float(np.quantile(within_iccs, .75)), 4),
    "frac_positive": round(float((within_iccs > 0).mean()), 4),
    "raw_group_icc_median_ref": 0.132,
    "note": "ICC across classrooms sharing the SAME quiz template; substantial value => classroom clustering is net of quiz template.",
}
print("  ", json.dumps(rob["m9_group_within_quiz"], indent=2))

# ---------- A9b latent logistic ICC at frozen 300 items ----------
print("[A9b] latent logistic ICC at 300 items ...")
try:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    lids = rng.choice(prim_ids, size=300, replace=False)
    subl = df[df["QuestionId"].isin(set(lids.tolist()))]
    lat, lin = [], []
    for q in lids:
        g = subl[subl["QuestionId"] == q][["IsCorrect", "GroupId"]]
        if g["GroupId"].nunique() < 10:
            continue
        try:
            m = BinomialBayesMixedGLM.from_formula("IsCorrect ~ 1", {"grp": "0 + C(GroupId)"}, g)
            r = m.fit_vb()
            sd = float(np.exp(r.vcp_mean[0]))
            lat.append(sd ** 2 / (sd ** 2 + np.pi ** 2 / 3))
            lin.append(float(atlas.loc[q, "ICC_raw"]))
        except Exception:
            continue
    rob["b_latent_logistic_icc"] = {"status": "ok", "n_items": len(lat),
                                    "latent_icc_med": float(np.median(lat)),
                                    "linear_icc_med_same_items": float(np.median(lin))}
except Exception as e:
    rob["b_latent_logistic_icc"] = {"status": "error", "msg": str(e)[:200]}
print("  ", json.dumps(rob["b_latent_logistic_icc"], indent=2))

with open(f"{OUT}/robust_results.json", "w") as f:
    json.dump(rob, f, indent=2)
print("DONE analysis_crossed_latent")

# -*- coding: utf-8 -*-
"""A6 : (1) analytic design-effect-corrected minimum-N table (page-one deliverable),
        (2) empirical split-half stability curves (iid vs whole-class sampling)."""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 20260711
SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")
OUT = "../results"
os.makedirs(OUT, exist_ok=True)

# DEFF used for the corrected table: empirically validated direct-resampling median (A4) as primary,
# ANOVA-formula median as the alternative column (both regenerated from source-of-truth files).
res = json.load(open(f"{OUT}/resample_results.json"))
core = json.load(open(f"{OUT}/core_headline.json"))
DEFF_boot = res["A4_ci_ratio"]["DEFF_boot_median"]        # ~2.9 (direct resampling)
DEFF_anova = core["H2_median_DEFF"]                        # ~3.6 (Kish formula)

# ---------- (1) corrected minimum-N table ----------
def iid_n(p, se):
    return p * (1 - p) / (se ** 2)

rows = []
for se in [0.02, 0.03, 0.05]:
    for p in [0.5, 0.65, 0.8]:
        n0 = iid_n(p, se)
        rows.append({
            "target_SE": se, "ref_p": p,
            "iid_N": int(np.ceil(n0)),
            "corrected_N_resampling": int(np.ceil(n0 * DEFF_boot)),
            "corrected_N_formula": int(np.ceil(n0 * DEFF_anova)),
            "extra_responses_resampling": int(np.ceil(n0 * DEFF_boot) - np.ceil(n0)),
        })
minN = pd.DataFrame(rows)
minN.to_csv(f"{OUT}/min_n_table.csv", index=False)
print("[A6.1] corrected minimum-N table (DEFF_resampling=%.2f, DEFF_formula=%.2f):" % (DEFF_boot, DEFF_anova))
print(minN.to_string(index=False))

# ---------- (2) empirical split-half stability ----------
print("\n[load] responses for split-half ...")
df = pq.read_table(SRC, columns=["QuestionId", "IsCorrect", "GroupId"]).to_pandas()
df = df.dropna(subset=["QuestionId", "GroupId", "IsCorrect"])
df["QuestionId"] = df["QuestionId"].astype(np.int32)
df["GroupId"] = df["GroupId"].astype(np.int32)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)

atlas = pd.read_csv(f"{OUT}/item_atlas_t12.csv").set_index("QuestionId")
# consistent high-volume item set so the two arms are comparable across budgets
hv = atlas[(atlas["N"] >= 2000) & (atlas["k"] >= 60)].index.values
rng = np.random.default_rng(SEED + 6)
if len(hv) > 3000:
    hv = rng.choice(hv, 3000, replace=False)
print("  high-volume items for split-half:", len(hv))

# preload per-item arrays
sub_all = df[df["QuestionId"].isin(set(hv.tolist()))]
grp = {q: (g["IsCorrect"].values.astype(np.int8), g["GroupId"].values) for q, g in sub_all.groupby("QuestionId")}

budgets = [50, 100, 200, 400, 800]
def stability_curve(design, reps=40):
    out = {}
    for n in budgets:
        r_vals = []
        for rep in range(reps):
            estA, estB = [], []
            for q in hv:
                y, gid = grp[int(q)]
                N = len(y)
                if design == "iid":
                    if N < 2 * n:  # need two disjoint samples of n
                        continue
                    perm = rng.permutation(N)
                    a = y[perm[:n]].mean(); b = y[perm[n:2 * n]].mean()
                else:  # cluster: split groups into two disjoint halves, draw whole groups to ~n each
                    uniq = np.unique(gid)
                    if len(uniq) < 4: continue
                    gperm = rng.permutation(uniq)
                    half = len(gperm) // 2
                    gA, gB = set(gperm[:half].tolist()), set(gperm[half:].tolist())
                    # accumulate whole groups until >= n
                    def draw(gset):
                        acc = 0; vals = []
                        order = rng.permutation(list(gset))
                        for gg in order:
                            m = y[gid == gg]
                            vals.append(m); acc += len(m)
                            if acc >= n: break
                        return np.concatenate(vals) if vals else None
                    va, vb = draw(gA), draw(gB)
                    if va is None or vb is None or len(va) < n or len(vb) < n: continue
                    a = va.mean(); b = vb.mean()
                estA.append(a); estB.append(b)
            if len(estA) > 30:
                r_vals.append(np.corrcoef(estA, estB)[0, 1])
        out[n] = {"stability": float(np.mean(r_vals)), "sd": float(np.std(r_vals)), "n_items_used": len(estA), "reps": len(r_vals)}
    return out

print("[A6.2] split-half stability (iid) ...")
iid_curve = stability_curve("iid", reps=30)
print("  ", json.dumps(iid_curve))
print("[A6.2] split-half stability (cluster/whole-class) ...")
clu_curve = stability_curve("cluster", reps=30)
print("  ", json.dumps(clu_curve))

# n-gap to reach stability 0.9 (interpolate)
def n_for_target(curve, target=0.9):
    xs = np.array(budgets, float)
    ys = np.array([curve[n]["stability"] for n in budgets])
    if ys.max() < target: return None
    for i in range(1, len(xs)):
        if ys[i] >= target and ys[i - 1] < target:
            frac = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return float(xs[0]) if ys[0] >= target else None

n_iid = n_for_target(iid_curve)
n_clu = n_for_target(clu_curve)
a6 = {
    "DEFF_resampling": DEFF_boot, "DEFF_formula": DEFF_anova,
    "min_n_table": rows,
    "split_half_iid": iid_curve, "split_half_cluster": clu_curve,
    "n_reach_0.9_iid": n_iid, "n_reach_0.9_cluster": n_clu,
    "empirical_n_ratio_cluster_over_iid": (n_clu / n_iid) if (n_iid and n_clu) else None,
    "n_items_split_half": int(len(hv)),
}
with open(f"{OUT}/minN_results.json", "w") as f:
    json.dump(a6, f, indent=2)
print("\n[A6] n to reach stability 0.9: iid=%s cluster=%s ratio=%s" %
      (n_iid, n_clu, a6["empirical_n_ratio_cluster_over_iid"]))
print("DONE analysis_minN")

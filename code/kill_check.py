# -*- coding: utf-8 -*-
"""Kill-check for the clustered-item-analytics (design-effects) study.

Core question: within an item, do multiple students share the same GroupId
(classroom)? If cluster sizes m are ~1, DEFF = 1 + (m-1)*ICC ~= 1 and the whole
design-effects argument collapses. Also compute a first ICC read for a handful
of high-volume items to confirm nonzero within-class correlation of IsCorrect.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = os.environ.get("EEDI_T12", "../data/resp_t12.parquet")

print("loading columns QuestionId, UserId, IsCorrect, GroupId ...")
df = pq.read_table(SRC, columns=["QuestionId", "UserId", "IsCorrect", "GroupId"]).to_pandas()
print("raw rows:", len(df))

# basic hygiene: IsCorrect in {0,1}; GroupId, QuestionId present
print("IsCorrect values:", df["IsCorrect"].value_counts(dropna=False).to_dict())
print("GroupId null:", int(df["GroupId"].isna().sum()), " QuestionId null:", int(df["QuestionId"].isna().sum()))
df = df.dropna(subset=["GroupId", "QuestionId", "IsCorrect"])
df["GroupId"] = df["GroupId"].astype(np.int64)
df["QuestionId"] = df["QuestionId"].astype(np.int64)
df["IsCorrect"] = df["IsCorrect"].astype(np.int8)

n_resp = len(df)
n_q = df["QuestionId"].nunique()
n_g = df["GroupId"].nunique()
n_s = df["UserId"].nunique()
print(f"\nclean: {n_resp:,} responses | {n_q:,} questions | {n_g:,} groups | {n_s:,} students")
print("overall correct rate:", round(df["IsCorrect"].mean(), 4))

# ---- cluster structure: responses per (QuestionId, GroupId) ----
qg = df.groupby(["QuestionId", "GroupId"], sort=False).size()
print("\n== responses per (question, group) cluster ==")
print("n item-group cells:", f"{len(qg):,}")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  m quantile {q}: {qg.quantile(q):.1f}")
print("  m mean:", round(qg.mean(), 2), " max:", int(qg.max()))
print("  share of cells with m==1:", round((qg == 1).mean(), 4))
print("  share of cells with m>=5:", round((qg >= 5).mean(), 4))
print("  share of cells with m>=10:", round((qg >= 10).mean(), 4))

# ---- per-item: groups per item, mean cluster size, responses ----
per_item = df.groupby("QuestionId").agg(
    n=("IsCorrect", "size"),
    n_groups=("GroupId", "nunique"),
)
per_item["mbar"] = per_item["n"] / per_item["n_groups"]
print("\n== per-item summary ==")
print("median responses/item:", int(per_item["n"].median()))
print("median groups/item:", int(per_item["n_groups"].median()))
print("median mean-cluster-size mbar:", round(per_item["mbar"].median(), 2))
print("items with >=200 resp AND mbar>=5:", int(((per_item["n"] >= 200) & (per_item["mbar"] >= 5)).sum()))

# ---- quick ICC read on the 30 highest-volume items ----
def item_icc(sub):
    # one-way random-effects ANOVA ICC(1) on binary IsCorrect within GroupId
    grp = sub.groupby("GroupId")["IsCorrect"]
    ni = grp.size().values
    yi = grp.mean().values
    N = ni.sum()
    k = len(ni)
    if k < 2:
        return np.nan, np.nan, k
    ybar = sub["IsCorrect"].mean()
    ssb = np.sum(ni * (yi - ybar) ** 2)
    # within
    ssw = np.sum((sub["IsCorrect"].values - sub.groupby("GroupId")["IsCorrect"].transform("mean").values) ** 2)
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    m0 = (N - np.sum(ni ** 2) / N) / (k - 1)  # design-adjusted average cluster size
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    return icc, m0, k

top = per_item.sort_values("n", ascending=False).head(30).index
print("\n== ICC on 30 highest-volume items ==")
iccs = []
for q in top:
    sub = df[df["QuestionId"] == q]
    icc, m0, k = item_icc(sub)
    iccs.append(icc)
    deff = 1 + (m0 - 1) * icc if not np.isnan(icc) else np.nan
    print(f"  Q{q}: n={len(sub):,} groups={k} m0={m0:.1f} ICC={icc:.3f} DEFF={deff:.2f}")
iccs = np.array([x for x in iccs if not np.isnan(x)])
print("\nICC (top30) median:", round(np.median(iccs), 3), " mean:", round(np.mean(iccs), 3),
      " min:", round(iccs.min(), 3), " max:", round(iccs.max(), 3))
print("\nKILL-CHECK VERDICT INPUTS ABOVE. Need: mbar meaningfully >1 for many items AND ICC>0.")

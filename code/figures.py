# -*- coding: utf-8 -*-
"""Publication-grade figures (300 dpi, colorblind-safe)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../results"
FIG = "../results/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 120})
# Okabe-Ito colorblind-safe
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "gray": "#555555"}

atlas = pd.read_csv(f"{OUT}/item_atlas_t12.csv")
R = json.load(open(f"{OUT}/RESULTS.json"))

# ---------- Figure 1: design-effect atlas (ICC + DEFF distributions) ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].hist(atlas["ICC_raw"], bins=60, color=CB["blue"], alpha=0.85, edgecolor="white", linewidth=0.3)
m_icc = R["H1_icc"]["median"]
ax[0].axvline(m_icc, color=CB["red"], lw=2, label=f"median = {m_icc:.3f}")
ax[0].axvline(0.05, color=CB["gray"], lw=1.2, ls="--", label="0.05 reference")
ax[0].set_xlabel("Item-level ICC of correctness (within classroom)")
ax[0].set_ylabel("Number of items")
ax[0].set_title("(a) Intraclass correlation across 15,717 items")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_xlim(-0.02, 0.45)

deff = atlas["DEFF"].clip(upper=12)
ax[1].hist(deff, bins=60, color=CB["green"], alpha=0.85, edgecolor="white", linewidth=0.3)
m_deff = R["H2_deff_formula"]["median"]
ax[1].axvline(m_deff, color=CB["red"], lw=2, label=f"median (formula) = {m_deff:.2f}")
ax[1].axvline(R["deff_resampling"]["DEFF_bootstrap_median"], color=CB["orange"], lw=2, ls="-.",
             label=f"median (resampling) = {R['deff_resampling']['DEFF_bootstrap_median']:.2f}")
ax[1].axvline(1.0, color=CB["gray"], lw=1.2, ls="--", label="no clustering (=1)")
ax[1].set_xlabel("Design effect on item difficulty  (clipped at 12)")
ax[1].set_ylabel("Number of items")
ax[1].set_title("(b) Design effects across items")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_atlas.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 2: heterogeneity by subject and difficulty ----------
subj = pd.DataFrame(R["heterogeneity"]["by_subject"])
subj = subj[subj["n_items"] >= 50].sort_values("deff_med")
band = pd.DataFrame(R["heterogeneity"]["by_difficulty"])
band_order = ["hard(.02-.4)", "medium(.4-.6)", "easy(.6-.8)", "very_easy(.8-.98)"]
band["diff_band"] = pd.Categorical(band["diff_band"], band_order, ordered=True)
band = band.sort_values("diff_band")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].barh(subj["subject"], subj["deff_med"], color=CB["blue"], alpha=0.85)
ax[0].axvline(m_deff, color=CB["red"], lw=1.5, ls="--", label=f"overall median {m_deff:.2f}")
ax[0].set_xlabel("Median design effect (item difficulty)")
ax[0].set_title("(a) By subject")
ax[0].legend(frameon=False, fontsize=8)
for i, (_, r) in enumerate(subj.iterrows()):
    ax[0].text(r["deff_med"] + 0.03, i, f"n={int(r['n_items'])}", va="center", fontsize=7, color=CB["gray"])

ax[1].bar(range(len(band)), band["deff_med"], color=CB["green"], alpha=0.85)
ax[1].set_xticks(range(len(band)))
ax[1].set_xticklabels(["hard\n.02-.4", "medium\n.4-.6", "easy\n.6-.8", "v.easy\n.8-.98"], fontsize=8)
ax[1].axhline(m_deff, color=CB["red"], lw=1.5, ls="--")
ax[1].set_ylabel("Median design effect")
ax[1].set_title("(b) By item difficulty band")
ax[1].set_ylim(0, max(band["deff_med"]) * 1.25)
for i, v in enumerate(band["deff_med"]):
    ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_heterogeneity.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: coverage of naive vs cluster 95% CI (the certification) ----------
cov = R["H3_coverage"]
ns = sorted(int(k) for k in cov)
naive = [cov[str(n)]["naive_coverage"] for n in ns]
clu = [cov[str(n)]["cluster_coverage"] for n in ns]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(ns, naive, "o-", color=CB["red"], lw=2, label="Naive binomial 95% CI (i.i.d. assumption)")
ax.plot(ns, clu, "s-", color=CB["blue"], lw=2, label="Cluster-robust 95% CI")
ax.axhline(0.95, color=CB["gray"], lw=1.3, ls="--", label="Nominal 95%")
ax.set_xlabel("Responses drawn (whole-classroom sampling)")
ax.set_ylabel("Empirical coverage of true item difficulty")
ax.set_title("Naive intervals under-cover on clustered platform data")
ax.set_ylim(0.6, 1.0); ax.set_xticks(ns)
ax.legend(frameon=False, fontsize=8, loc="center right")
for n, v in zip(ns, naive):
    ax.text(n, v + 0.012, f"{v:.2f}", ha="center", fontsize=8, color=CB["red"])
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 4: split-half stability curves ----------
sh = R["split_half"]
ns2 = sorted(int(k) for k in sh["iid"])
iid = [sh["iid"][str(n)]["stability"] for n in ns2]
clu2 = [sh["cluster"][str(n)]["stability"] for n in ns2]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(ns2, iid, "o-", color=CB["blue"], lw=2, label="Sampling individual responses (i.i.d.)")
ax.plot(ns2, clu2, "s-", color=CB["orange"], lw=2, label="Sampling whole classrooms (realistic)")
ax.axhline(0.9, color=CB["gray"], lw=1.2, ls="--", label="stability = 0.90")
ax.axvline(sh["n_reach_0.9_iid"], color=CB["blue"], lw=1, ls=":")
ax.axvline(sh["n_reach_0.9_cluster"], color=CB["orange"], lw=1, ls=":")
ax.set_xscale("log")
ax.set_xlabel("Responses per estimate (log scale)")
ax.set_ylabel("Split-half stability of item difficulty")
ax.set_title(f"To reach 0.90: {sh['n_reach_0.9_iid']:.0f} vs {sh['n_reach_0.9_cluster']:.0f} responses")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_splithalf.png", dpi=300, bbox_inches="tight"); plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f)

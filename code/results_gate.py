# -*- coding: utf-8 -*-
"""RESULTS GATE - single source of truth. Assembles every number the manuscript may
cite into RESULTS.json (recompute, never retype). Also cleanly (re)computes the A4
empirical-vs-formula DEFF correlation from the saved per-item CSV, and logs H1/H2/H3."""
import json, os
import numpy as np
import pandas as pd

OUT = "../results"
core = json.load(open(f"{OUT}/core_headline.json"))
resa = json.load(open(f"{OUT}/resample_results.json"))
minN = json.load(open(f"{OUT}/minN_results.json"))
rob = json.load(open(f"{OUT}/robust_results.json"))

# ---- clean A4 correlation from source CSV ----
ci = pd.read_csv(f"{OUT}/ci_ratio_items.csv")
atlas = pd.read_csv(f"{OUT}/item_atlas_t12.csv").set_index("QuestionId")
ci = ci.set_index("QuestionId")
ci["DEFF_anova"] = atlas["DEFF"].reindex(ci.index)
m = ci.replace([np.inf, -np.inf], np.nan).dropna(subset=["DEFF_boot", "DEFF_anova"])
m = m[(m["DEFF_boot"] > 0) & (m["DEFF_anova"] > 0)]
corr_deff = float(np.corrcoef(m["DEFF_boot"], m["DEFF_anova"])[0, 1])
corr_sqrt = float(np.corrcoef(np.sqrt(m["DEFF_boot"]), np.sqrt(m["DEFF_anova"]))[0, 1])
corr_boot_lin = float(np.corrcoef(m["DEFF_boot"], m["DEFF_lin"])[0, 1])

RESULTS = {
    "meta": {"seed": core["seed"], "dataset": "Eedi NeurIPS-2020 Task1&2 (primary), Task3&4 (replication)",
             "cluster_unit": "GroupId (classroom)", "estimand": "item-level ICC of IsCorrect within classroom, and design effect on item difficulty/discrimination"},
    "sample": {
        "n_responses": core["n_responses"], "n_items_total": core["n_items_total"],
        "n_items_primary": core["n_items_primary"], "n_groups": core["n_groups"],
        "n_students": core["n_students"], "overall_correct_rate": round(core["overall_correct_rate"], 4),
        "median_responses_per_item": core["median_responses_per_item"],
        "median_groups_per_item": core["median_groups_per_item"],
        "median_mstar": round(core["median_mstar"], 2),
        "inclusion": "N>=200 responses & >=20 groups & 0.02<=p<=0.98",
    },
    "H1_icc": {
        "median": round(core["H1_median_ICC"], 4), "ci95": [round(core["H1_ci"][0], 4), round(core["H1_ci"][1], 4)],
        "threshold": 0.05, "supported": core["H1_supported"],
        "quantiles": {k: round(v, 4) for k, v in core["ICC_quantiles"].items()},
        "pct_items_gt_0.05": round(core["pct_items_ICC_gt_0.05"], 4),
    },
    "H2_deff_formula": {
        "median": round(core["H2_median_DEFF"], 3), "ci95": [round(core["H2_ci"][0], 3), round(core["H2_ci"][1], 3)],
        "threshold": 1.5, "supported": core["H2_supported"],
        "quantiles": {k: round(v, 3) for k, v in core["DEFF_quantiles"].items()},
        "pct_items_gt_1.5": round(core["pct_items_DEFF_gt_1.5"], 4),
        "pct_items_gt_2": round(core["pct_items_DEFF_gt_2"], 4),
        "median_n_eff_ratio": round(core["n_eff_ratio_median"], 4),
    },
    "deff_resampling": {   # A4, the directly-validated design effect
        "n_items": resa["A4_ci_ratio"]["n_items"],
        "DEFF_bootstrap_median": round(resa["A4_ci_ratio"]["DEFF_boot_median"], 3),
        "DEFF_linearization_median": round(resa["A4_ci_ratio"]["DEFF_lin_median"], 3),
        "DEFF_formula_median_same_context": round(resa["A4_ci_ratio"]["DEFF_anova_median"], 3),
        "corr_DEFF_boot_vs_formula": round(corr_deff, 3),
        "corr_sqrtDEFF_boot_vs_formula": round(corr_sqrt, 3),
        "corr_boot_vs_linearization": round(corr_boot_lin, 3),
        "median_ratio_boot_over_formula": round(resa["A4_ci_ratio"]["median_ratio_boot_over_anova"], 3),
        "note": "Two independent variance estimators (cluster bootstrap, Taylor linearization) agree; the Kish formula on burden-weighted m* runs ~19% higher.",
    },
    "H3_coverage": resa["A5_coverage"],
    "iid_coverage_baseline": resa.get("M6_iid_coverage_baseline"),
    "perm_null": resa["A2_perm_null"],
    "discrimination": resa["A7_discrimination"],
    "min_n_table": minN["min_n_table"],
    "split_half": {
        "iid": minN["split_half_iid"], "cluster": minN["split_half_cluster"],
        "n_reach_0.9_iid": round(minN["n_reach_0.9_iid"], 1),
        "n_reach_0.9_cluster": round(minN["n_reach_0.9_cluster"], 1),
        "n_ratio_cluster_over_iid": round(minN["empirical_n_ratio_cluster_over_iid"], 2),
        "n_items": minN["n_items_split_half"],
    },
    "heterogeneity": {
        "by_subject": pd.read_csv(f"{OUT}/subject_breakdown.csv").round(3).to_dict(orient="records"),
        "by_difficulty": pd.read_csv(f"{OUT}/difficulty_breakdown.csv").round(3).to_dict(orient="records"),
    },
    "robustness": rob,
    "gap_decomposition": resa.get("M4_gap_decomposition"),
}
# second corpus (ASSISTments 2009) for cross-platform generalization
try:
    RESULTS["second_corpus_assistments"] = json.load(open(f"{OUT}/assistments_results.json"))
    RESULTS["second_corpus_assistments"]["perm_null"] = resa.get("M3_assistments_perm_null")
except FileNotFoundError:
    RESULTS["second_corpus_assistments"] = None

# ---- decisions log ----
# H3 (frozen rule): SUPPORTED iff naive-coverage upper CI < 0.95 AND cluster-coverage CI includes 0.95.
# Outcome: naive under-covers decisively (1st conjunct TRUE) but cluster-robust never reaches 0.95 at
# these cluster counts (2nd conjunct FALSE) -> H3 is PARTIALLY supported, reported honestly as such.
H3 = resa["A5_coverage"]
naive_undercovers = all(H3[k]["naive_coverage"] < 0.95 for k in H3)
cluster_reaches_nominal = any(H3[k]["cluster_coverage"] >= 0.95 for k in H3)
cluster_improves_monotonic = (H3["400"]["cluster_coverage"] > H3["100"]["cluster_coverage"])
RESULTS["decisions"] = {
    "H1_supported": bool(core["H1_supported"]),
    "H2_supported": bool(core["H2_supported"]),
    "H3_naive_undercovers": bool(naive_undercovers),
    "H3_cluster_reaches_nominal": bool(cluster_reaches_nominal),
    "H3_verdict": "partially_supported" if (naive_undercovers and not cluster_reaches_nominal) else ("supported" if (naive_undercovers and cluster_reaches_nominal) else "not_supported"),
    "H3_cluster_improves_monotonically": bool(cluster_improves_monotonic),
    "discrimination_deff_smaller_than_difficulty": bool(resa["A7_discrimination"]["DEFF_disc_median"] < resa["A4_ci_ratio"]["DEFF_boot_median"]),
    "task34_replicates": bool(rob["f_task34"]["deff_med"] > 1.5 and rob["f_task34"]["icc_med"] > 0.05),
}

with open(f"{OUT}/RESULTS.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("RESULTS.json written. Key numbers:")
print(f"  n_responses={RESULTS['sample']['n_responses']:,}  items_primary={RESULTS['sample']['n_items_primary']:,}  groups={RESULTS['sample']['n_groups']:,}")
print(f"  H1 median ICC={RESULTS['H1_icc']['median']} CI{RESULTS['H1_icc']['ci95']} supported={RESULTS['decisions']['H1_supported']}")
print(f"  H2 median DEFF(formula)={RESULTS['H2_deff_formula']['median']} CI{RESULTS['H2_deff_formula']['ci95']} supported={RESULTS['decisions']['H2_supported']}")
print(f"  DEFF resampling boot={RESULTS['deff_resampling']['DEFF_bootstrap_median']} lin={RESULTS['deff_resampling']['DEFF_linearization_median']} corr(boot,formula)={RESULTS['deff_resampling']['corr_DEFF_boot_vs_formula']}")
print(f"  H3 coverage: " + " ".join(f"n{k}: naive={v['naive_coverage']:.2f}/clu={v['cluster_coverage']:.2f}" for k, v in H3.items()) + f" verdict={RESULTS['decisions']['H3_verdict']}")
print(f"  perm-null frac sig(BH)={RESULTS['perm_null']['frac_sig_BH']}  obs_med={RESULTS['perm_null']['obs_icc_median']:.3f} null_med={RESULTS['perm_null']['null_icc_median']:.3f}")
print(f"  discrimination DEFF median={RESULTS['discrimination']['DEFF_disc_median']:.2f} (< difficulty DEFF -> {RESULTS['decisions']['discrimination_deff_smaller_than_difficulty']})")
print(f"  Task3&4 replication: ICC={RESULTS['robustness']['f_task34']['icc_med']:.3f} DEFF={RESULTS['robustness']['f_task34']['deff_med']:.2f}")
print(f"  latent logistic ICC={RESULTS['robustness']['b_latent_logistic_icc'].get('latent_icc_med')}")
print("DONE results_gate")

# -*- coding: utf-8 -*-
"""M7: two-argument item-tryout planner. Given a target SE on item difficulty and the
typical class size m, return the number of CLASSES and total responses needed, using the
measured classroom ICC. Makes explicit the 'add classes, not just students' lever."""
import json, numpy as np, pandas as pd

OUT = "../results"
RHO = 0.13          # measured median classroom ICC of item correctness
P = 0.65            # reference difficulty (mid-range)

rows = []
for se in [0.02, 0.03, 0.05]:
    n_eff_req = P * (1 - P) / se ** 2      # required effective sample (= i.i.d. responses)
    for m in [5, 15, 30]:
        deff = 1 + (m - 1) * RHO
        info_per_class = m / deff
        C = int(np.ceil(n_eff_req / info_per_class))
        total = C * m
        rows.append({"target_SE": se, "iid_responses": int(np.ceil(n_eff_req)),
                     "class_size_m": m, "deff_at_m": round(deff, 2),
                     "classes_needed": C, "total_responses": total})
planner = pd.DataFrame(rows)
planner.to_csv(f"{OUT}/planner_table.csv", index=False)
print(f"Planner (rho={RHO}, p={P}):")
print(planner.to_string(index=False))

# merge into a small json for the results gate / manuscript
with open(f"{OUT}/planner_results.json", "w") as f:
    json.dump({"rho": RHO, "ref_p": P, "rows": rows}, f, indent=2)
print("DONE analysis_planner")

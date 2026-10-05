# -*- coding: utf-8 -*-
"""Build the answer-level input tables from the Eedi NeurIPS 2020 Education Challenge zip.

For Task 1&2 and for Task 3&4: left-join the train file (QuestionId, UserId, AnswerId,
IsCorrect) with the answer metadata (DateAnswered, Confidence, GroupId, QuizId) on AnswerId,
drop AnswerId, and sort by UserId, then DateAnswered (stable sort). Missing GroupId/QuizId in
the metadata are coded -1. All rows are kept.

Several analyses draw random subsets of the responses inside a class, so the row order of
these tables is part of the input. Building them with this script gives the row order used
for the article.

Input:   EEDI_ZIP (default ../data/eedi_challenge.zip), the challenge data.zip (see data/README.md).
Outputs: EEDI_T12 (default ../data/resp_t12.parquet), 15,867,850 rows;
         EEDI_T34 (default ../data/resp_t34.parquet), 1,382,727 rows.
Run:     from code/:  python build_inputs.py
"""
import io, os, zipfile
import numpy as np
import pandas as pd

ZIP = os.environ.get("EEDI_ZIP", "../data/eedi_challenge.zip")
TARGETS = {
    "t12": (os.environ.get("EEDI_T12", "../data/resp_t12.parquet"),
            "data/train_data/train_task_1_2.csv", "data/metadata/answer_metadata_task_1_2.csv"),
    "t34": (os.environ.get("EEDI_T34", "../data/resp_t34.parquet"),
            "data/train_data/train_task_3_4.csv", "data/metadata/answer_metadata_task_3_4.csv"),
}


def read_member(z, member, **kw):
    with z.open(member) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), **kw)


z = zipfile.ZipFile(ZIP)
for task, (out, train_member, ans_member) in TARGETS.items():
    print(f"[{task}] reading {train_member} and {ans_member} ...", flush=True)
    tr = read_member(z, train_member,
                     usecols=["QuestionId", "UserId", "AnswerId", "IsCorrect"],
                     dtype={"QuestionId": np.int32, "UserId": np.int32,
                            "AnswerId": np.int64, "IsCorrect": np.int8})
    am = read_member(z, ans_member,
                     usecols=["AnswerId", "DateAnswered", "Confidence", "GroupId", "QuizId"],
                     dtype={"Confidence": np.float32})
    am = am[am["AnswerId"].notna()].copy()
    am["AnswerId"] = am["AnswerId"].astype(np.float64).astype(np.int64)
    am = am.drop_duplicates(subset="AnswerId", keep="first")
    am["DateAnswered"] = pd.to_datetime(am["DateAnswered"])
    am["GroupId"] = am["GroupId"].fillna(-1).astype(np.int32)
    am["QuizId"] = am["QuizId"].fillna(-1).astype(np.int32)
    df = tr.merge(am, on="AnswerId", how="left", validate="one_to_one")
    df = df.drop(columns=["AnswerId"]).sort_values(
        ["UserId", "DateAnswered"], kind="stable").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_parquet(out)
    print(f"[{task}] {len(df):,} rows -> {out}", flush=True)
print("DONE build_inputs")

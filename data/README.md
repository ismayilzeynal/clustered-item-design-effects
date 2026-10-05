# Data directory

The raw response data are NOT included here: the Eedi data are released for research use without row-level redistribution. Put the input files in this folder (the scripts read them from here by default), or set the environment variables below to their locations.

| File in `data/` | Environment variable | Used by | How to get it |
|---|---|---|---|
| `eedi_challenge.zip` | `EEDI_ZIP` | `build_inputs.py`, `analysis_core.py` | download, step 1 |
| `resp_t12.parquet` | `EEDI_T12` | every Eedi analysis | `build_inputs.py`, step 2 |
| `resp_t34.parquet` | `EEDI_T34` | `analysis_robust.py` | `build_inputs.py`, step 2 |
| `assistments09_skill_builder.csv` | `ASSIST09` | `analysis_assistments.py`, `analysis_r2.py` | download, step 3 |
| `subsamples.npz`, `student_totals.parquet` | - | later scripts | written by `analysis_core.py` |

## 1. Eedi NeurIPS 2020 Education Challenge data
Download the public challenge data (tasks 1-4), `data.zip`, from the organizers' distribution referenced in the challenge guide (Wang et al., 2020, arXiv:2007.12061):

    https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip

Save it as `data/eedi_challenge.zip`. The file used for the article has 656,787,242 bytes and SHA-256 `c7f01672360f1adeb3cf9507d72455d7be035bf897e4a167293e8938049800e1`. The data are released for research use (CC BY-NC-ND 4.0); cite Wang et al. (2020).

Members used:
- `data/train_data/train_task_1_2.csv` and `data/metadata/answer_metadata_task_1_2.csv` (Task 1&2, primary analysis)
- `data/train_data/train_task_3_4.csv` and `data/metadata/answer_metadata_task_3_4.csv` (Task 3&4, held-out replication)
- `data/metadata/question_metadata_task_1_2.csv` and `data/metadata/subject_metadata.csv` (read directly by `analysis_core.py` for subject labels)

## 2. Build the answer-level tables
From `code/`:

    python build_inputs.py

For each task it left-joins the train file (`QuestionId, UserId, AnswerId, IsCorrect`) with the answer metadata (`DateAnswered, Confidence, GroupId, QuizId`) on `AnswerId`, drops `AnswerId`, and sorts by `UserId`, then `DateAnswered` (stable sort). Several analyses draw random subsets of the responses inside a class, so this row order is part of the input.

| Output | Rows | SHA-256 with the pinned packages |
|---|---|---|
| `resp_t12.parquet` | 15,867,850 | `c27099e0ccd4072918085bfdea984a769753cdbc27c26ab38fada58759559ce5` |
| `resp_t34.parquet` | 1,382,727 | `49971f533a40f2a8b7fee769c22c931ca3085693f9eeed5d9d01c6d8dfd1aaab` |

These are the files used for the article; other package versions can write different bytes for the same table.

## 3. ASSISTments 2009-2010 skill-builder data (second corpus)
From the ASSISTments data site, 2009-2010 skill-builder page:

    https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

download the original `skill_builder_data.csv` (not the "corrected" or "collapsed" variants) and save it as `data/assistments09_skill_builder.csv`. The file used for the article has 83,201,940 bytes and SHA-256 `f22e3fb7872c1784ce93b0f9ebabbe0cbcac4f896fd8b4a11667b9715d77dbdc`. Cite Feng, Heffernan, and Koedinger (2009), https://doi.org/10.1007/s11257-009-9063-7.

## Generated files
`analysis_core.py` writes `subsamples.npz` (the frozen item subsamples used by the resampling, coverage and revision scripts) and `student_totals.parquet` (per-student totals for the discrimination analysis) into this folder.

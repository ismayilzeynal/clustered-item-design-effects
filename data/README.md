# Data directory

The raw Eedi response data is NOT included here (research-use license, no row-level redistribution).

## How to obtain and build the inputs
1. Download the Eedi NeurIPS 2020 Education Challenge data from the official challenge distribution (see Wang et al., 2020, arXiv:2007.12061, for the pointer). You need the `train_data` and `metadata` folders.
2. Build the answer-level parquet caches this pipeline expects:
   - `resp_t12.parquet` (Task 1&2): left-join `train_data/train_task_1_2.csv` (UserId, QuestionId, IsCorrect) with `metadata/answer_metadata_task_1_2.csv` (GroupId, QuizId, DateAnswered, Confidence) on `AnswerId`. Keep columns `QuestionId, UserId, IsCorrect, DateAnswered, Confidence, GroupId, QuizId`.
   - `resp_t34.parquet` (Task 3&4): same join with the Task 3&4 files.
   - Place the challenge zip (containing `data/metadata/question_metadata_task_1_2.csv` and `data/metadata/subject_metadata.csv`) as `eedi_challenge.zip`, or point `EEDI_ZIP` at it.
3. Place `resp_t12.parquet` and `resp_t34.parquet` in this `data/` folder, or set the environment variables `EEDI_T12`, `EEDI_T34`, `EEDI_ZIP` to their locations.

Expected sizes: `resp_t12.parquet` ~ 15,867,850 rows; `resp_t34.parquet` ~ 1,382,727 rows.

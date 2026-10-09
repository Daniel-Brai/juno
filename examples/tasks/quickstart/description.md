Each row has two measurements, `x1` and `x2`, and a binary `label`.

Train a model on `train.csv` and predict `label` for every row of `test.csv`.
Write the predictions to `submission.csv` with the columns `id,label`, in the
same format as `sample_submission.csv`.

Submissions are scored by accuracy on hidden test labels. Hold out part of
`train.csv` to estimate your accuracy before registering a submission.

Only the Python standard library is available.

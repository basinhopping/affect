# Recalling a model

Any model the project reports can be rebuilt and checked against its record. Today that is done by rerunning it;
at the end of the project the final models are also saved with their weights.

## What a run keeps

Every run folder, `artifacts/runs/<sweep>/<config>_<fold>_seed<seed>/`, holds:

| File | What it holds |
| --- | --- |
| `spec.json` | every setting of the run (data, features, augmentation, model, training), its fold and seed, and the configuration hash |
| `provenance.json` | the code's SHA-256, the package versions, the GPU, the data manifest's SHA-256 and the folds' hash |
| `metrics_val.json` | the validation scores, and per epoch the training loss and the validation scores |
| `predictions_val.csv`, `predictions_test.csv` | every utterance's label, prediction and class probabilities |
| `metrics_test.json` | the test scores, recall per class and the confusion matrix |

A record (`artifacts/selections/<name>/selection.json`) names the sweep, the configuration hash, the folds and the
seeds of a chosen model, and why it was chosen.

## Rerunning a record

1. Take the record's sweep file, `artifacts/sweeps/<sweep>/sweep_file.json`. It holds the exact settings, folds and
   seeds of every job.
2. Check that the code is the version in the runs' `provenance.json` (`code_sha256`; `pushed_code_version` names the
   code archive the run came from). This repository's history holds the code at each release, not every version a
   run used: the runs reported on 2026-11-09 came from code archives of late October and early November 2026, all
   older than that release.
3. Copy the sweep file, give the copy a new `sweep_id`, and run its jobs: `python -m affect.cli enqueue --sweep-file
   <copy>`, then `python -m affect.cli worker` (on the GPU node, `scripts/smoke_enqueue.sh` and `scripts/worker.sh`).
4. Compare the new runs with the old ones:
   `python -m affect.cli compare-runs --sweep <new sweep_id> --differences-shown 3`. It compares each run with every
   copy of the same configuration, fold and seed: the validation predictions byte for byte, the validation metrics
   exactly, and the two test files by SHA-256. It ends SAME, DIFFERENT or INCOMPLETE.

Training is deterministic on the same code and machine: every rerun so far repeated its predictions exactly. The film
runs of the week of 2026-11-09 are such a check at scale, since each film run trains its refit again from the start
and its IEMOCAP predictions are compared with the refit's run by SHA-256.

## Saved weights

The runs keep no weights; a model is rebuilt by rerunning it. Once the set of models for the paper is fixed, each
final record is rerun with its weights saved (4 bytes a parameter; the reported models have 0.13 to 3.74 million
parameters, listed in [models.md](models.md)), and a recall command loads a saved model,
predicts the test utterances again and checks that it reproduces the run's `predictions_test.csv` exactly.

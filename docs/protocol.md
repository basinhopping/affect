# Protocol and controls

Every run of the project follows the rules on this page. They are also the controls my project partner uses, so
that the same data, folds and seeds give the same splits and the same random draws on any machine. Each control
can be checked by a hash that every run records in its `provenance.json`.

## Data

IEMOCAP, the improvised and scripted dialogs of its five sessions. IEMOCAP is licensed and is not in this
repository: request it from its owners at USC SAIL (https://sail.usc.edu/iemocap/), then build the project's copy
with `python -m affect.cli ingest` and `python -m affect.cli assemble` (see the README).

| Class set | Classes | Utterances |
| --- | --- | --- |
| Four classes (the main set) | neutral 1,708; happy with excited 1,636; angry 1,103; sad 1,084 | 5,531 |
| Three classes (for the cross-corpus films) | happy with excited, angry, sad | 3,823 |

Each label is IEMOCAP's own majority label, with excited merged into happy. The assembled copy's manifest has
SHA-256 `d6c6aefc...`, recorded in every run as `data_manifest_sha256`.

## Folds

The split schemes are in `src/affect/data/iemocap_splits.py`. Each run records the hash of its split's folds as
`folds_hash`.

| Protocol | Folds | What it holds out | Folds hash |
| --- | --- | --- | --- |
| Session-specific (leave one session out) | 5 | fold k tests session k, validates on session k + 1 (session 1 for fold 5), trains on the other three | `13d21ab6` four classes, `a2f6dd38` three |
| Speaker-specific (leave one speaker out) | 10 | one speaker tests, the other speaker of that session validates, the eight speakers of the other sessions train | `28d0bbb5` |
| Session-agnostic (utterance folds) | 5 | utterances dealt into five groups by session, speaker and class; test speakers are also heard in training | `7da94e15` four classes, `95f824d3` three |
| Script held out | 15 | each fold tests one session's utterances of one script and keeps that script out of training; the control keeps another script out instead | |

One script fold, session 5 script 3, holds no sad utterance, so macro accuracy is not defined on it and the script
comparisons read the other 14 folds.

## Seeds

Screens use seeds 1, 2 and 3. Confirmations and every reported record use seeds 6 to 10, so no reported number
comes from a seed that chose it. Smoke tests use seed 1. A seed fixes the first weights, the training crops and the
batch order; it is in each run's `spec.json` and in its run ID.

## Choosing and the refit

Every setting is chosen on validation only, by mean validation macro accuracy over the folds and seeds. A screen
tries a base and 30 random configurations of a search space; the two best are confirmed on new seeds; the better of
the two is the record. The record is then refitted: retrained on training plus validation (four sessions, or nine
speakers) for a fixed number of epochs, the median of the epochs its validation runs chose, and tested once. Only
refits are reported. An SVM keeps the C and gamma it chose on validation. The refit is recorded in `spec.json` as
`train.refit` and `refit_epochs`.

## Scores

Macro accuracy, the mean of the per-class recalls (called UA in the code and in older records), is the main score.
Macro-F1 and accuracy (WA) are reported beside it, with the recall of each class. Every number is recomputed from a
run's `predictions_test.csv` and checked against its `metrics_test.json`, then rounded half up (macro accuracy to
4 decimals, differences to 2 decimals of a point).

## Statistics

Before any test file of a set of runs is read, a rule file fixes which records are compared and how, with its UTC
time and SHA-256. A record's score is the mean over its seeds of the mean over its folds. Two records are compared
with a paired bootstrap of 2,000 draws over each fold's test utterances and over the seeds; the result is the
difference with its 95% interval and Holm's correction within each family of comparisons. A difference is better or
worse only when Holm's p is under 0.05, otherwise it is reported as no detectable difference. Two records are called
the same only when the whole interval lies within 2 points. Every comparison is repeated with whole dialogs drawn
together, as a check that the intervals do not lean on utterances of one dialog being independent.

## Code and machine

One code version per week of runs, its SHA-256 in every `provenance.json` (`code_sha256`). The runs used torch 2.14.0
on NVIDIA H100 NVL GPUs. On the same code and machine a run repeats exactly; on another machine small numerical
differences are possible, so shared runs are compared prediction by prediction against an agreed tolerance.

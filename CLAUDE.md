# CLAUDE.md

Guidance for Claude working in this repository (Claude Code, or Cowork with this folder connected). README.md describes the layout and every command in full; this file says how the work is done. Follow these rules without being asked, and say so when a request would break one.

## The project

Speech emotion recognition on IEMOCAP for the F26 Affective Computing project: a model hears one utterance and names its emotion. The standard label set is four classes (neutral, happy+excited, angry, sad; 5,531 utterances). The main protocol is session out: five folds, fold k tests session k, validates on session k + 1 (session 5 on session 1) and trains on the other three, so test speakers are never in training. The main metric is unweighted accuracy (UA), the mean of the per-class recalls.

## Who does what

- Training runs on a shared GPU server under each person's own login: the code in `~/affect`, the Python environment in `~/venvs/affect`, downloads in `~/cache`. Nothing uses `/scratch`.
- Claude cannot log in to the server. Claude prepares the files and the exact commands (no placeholders) and checks each before handing it over; the person runs each command and pastes the output; Claude compares it with what it should print before giving the next line. A line that prints something unexpected stops the sequence until it is understood.
- This repository is public. It never holds a login, a host name or a password. A PC that pushes code keeps the server login in `node.local.txt`, which git ignores.

## The shared GPU server

- On weekdays, GPU jobs run in the evening and overnight only. Every night's start line sets a stop time that writes `~/affect/queue/STOP`.
- Use only GPUs that `nvidia-smi` shows idle, and agree on GPUs with anyone else running that night.
- Work runs in tmux sessions that end by themselves (`gpu0` to `gpu3` for workers, one named session for a night's schedule). Never start anything that restarts itself or runs forever in the background. Never touch a tmux session someone else started.
- Never change the code in `~/affect` while workers run: a worker loads some modules late and would mix old and new code. Stop them (`touch ~/affect/queue/STOP`), wait until `tmux ls` lists no gpu session, then update. Remove `queue/STOP` before starting workers again, because a worker exits at once while it exists.

## Results: what makes a number trustworthy

- Every number reported comes from a file the code wrote, rounded half up from the exact value. Nothing is estimated or recalled.
- Test, never assume. Check a new sweep file with the repository's sweep expansion (`affect.catalog.load_all()`, then `affect.sweep.design.expand`) and `tests/test_sweep.py`; smoke-test before queueing (`scripts/smoke_enqueue.sh`); try a new command on stand-in data first; check every file copied between machines by SHA-256.
- Choices use validation only: `analyze` compares configurations on validation files, `select` records the choice (it never overwrites a record), and only then does `test-report` read the test files of the selected runs. A test number never chooses anything.
- Before any test result exists, write a RULE file in the night's prep folder: which runs are made, which test files will be read, which comparisons will be made (families of comparisons, paired bootstrap over seeds and test utterances, Holm's correction within each family), what happens to unfinished rows, and the UTC time it was fixed. It does not change once a test file has been read.
- Five seeds for a reported result, three for screening many configurations, one only for learning and smoke tests.
- Report a difference with its interval and Holm-adjusted p. A difference that is not detectable is reported as not detectable, never as "slightly better".

## Records

- `artifacts/analyses/nightN_prep_<date>/` holds a night's sweep-making script, RULE file, exact lines and a README. `artifacts/analyses/nightN_<date>/` holds the morning's scripts, outputs and a README saying what was run, what was read when, and what it found.
- Run folders (`artifacts/runs/<sweep_id>/<run_id>/`) are never overwritten or deleted. A failed job's folder is kept as evidence.
- `data/`, `artifacts/`, `queue/`, `logs/`, `HANDOFF.md`, `VERIFICATION_LOG.md` and `node.local.txt` stay out of git (see `.gitignore`). IEMOCAP is licensed: no audio, labels or transcripts go into the repository.

## A first run for someone new

`sweeps/intro_base_four.json`, `intro_base_fru5.json`, `intro_base_fru4.json` and `intro_base_no_neu3.json` repeat the seed-1 runs of the baseline CNN under the four class sets the project compares (5 jobs each, about a minute a job). The expected test UA for seed 1 is 0.5520, 0.4345, 0.5167 and 0.6873 (validation 0.5792, 0.4728, 0.5486 and 0.7235) when the package versions match those in the runs' `provenance.json`.

## Writing for the advisor

Documents for the advisor are written in the first person singular, in plain prose, with every number traceable to a file and no dashes as punctuation. Say what a result shows and what it does not, and say so when something was not tested.

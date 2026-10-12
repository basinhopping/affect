# affect

Speech emotion recognition on IEMOCAP for the F26 Affective Computing project. One canonical
tree: `C:\Affective Computing\affect` on the PC. The GPU node holds a working copy at
`~/affect`, replaced wholesale by each push. The plan, with every decision
and its reason, is the project document "Affective Computing Plan and Data Map".

`HANDOFF.md` (the current state) and `VERIFICATION_LOG.md` (every data check) travel with the code to the
node but stay out of the git repository, as do `data/` (IEMOCAP is licensed) and `artifacts/` (results).
`notebooks/cnn_3block.ipynb` walks one experiment from its settings to its results.

## Layout

```
affect/
  README.md  HANDOFF.md  VERIFICATION_LOG.md  findings.csv  requirements.in
  src/affect/          all code (the package)
    __init__.py        the node's CUDA MPS fix, applied on import
    settings.py        settings: frozen dataclasses, no defaults, checked both ways, hashed
    registry.py        @register_dataset / @register_feature / @register_model
    catalog.py         one import line per component module
    data/              iemocap.py, iemocap_splits.py (the split schemes), splits.py, ingest.py,
                       assemble.py, transfer.py, walk.py, manifest.py, audio.py, base.py
    eval/              metrics.py (UA, WA, F1), stats.py (t intervals, paired bootstrap, P(A>B), Holm)
    runs/              store.py (output folders), provenance.py
    env.py             environment and GPU checks
    cli.py             python -m affect.cli ...
    features/logmel.py
    augment/           settings.py (every augmentation setting), specaugment.py, speed.py (speed
                       perturbation), mixup.py
    models/            cnn.py (the CNN and the shared convolutional front end: normalization layer
                       and activation are settings), cnn_transformer.py, cnn_lif.py (the CNN front end
                       with LIF layers, the hybrid), snn.py (fully spiking), lif.py (the spiking
                       layers), lif_reference.py (the neuron in numpy, for the tests), heads.py (the
                       gender head)
    train/             settings.py, data.py (audio in memory, crops, speed-perturbed crops,
                       evaluation segments), loop.py (one job; train.loop.Step is one training step)
    runs/spec.py       a job's five settings sections, validated and hashed
    classical/         openSMILE functionals with an SVM or a TPOT search (own environment):
                       settings.py, functionals.py, methods.py, runner.py, selftest.py
    sweep/             design.py (expand a sweep file), enqueue.py, queue.py, worker.py,
                       status.py, smoke.py
    analysis/          validation.py: a finished sweep's configurations against its baseline (analyze);
                       groups.py: results by speaker gender and kind, the gender-routed system;
                       selection.py: select a configuration, then read its test results (test-report)
  tests/               python -m unittest discover -s tests -t .
  scripts/             node: common.sh setup_env.sh check.sh unpack.sh pack.sh worker.sh affect.sh
                             smoke_enqueue.sh (smoke-test, then enqueue, sweeps in order) and
                             gpu_schedule.sh (the weekend schedule of 2026-10-03 to 10-05), both 2026-10-03
                       PC:   push.ps1 pull.ps1 (the node's login is read from node.local.txt beside README.md, not in git)
  sweeps/              sweep files (JSON): base configuration, folds, stages
  notebooks/           cnn_3block.ipynb: settings, check, sweep and results of the 3-block CNN
  data/                not synced as code; see data/README.md
    iemocap/raw/         verified copies of the received folders, never edited
    iemocap/manifests/   one manifest per raw folder (path, bytes, SHA-256)
    iemocap/assembled/   what training reads, with MANIFEST.csv and ASSEMBLY.json
  artifacts/           results, written on the node and pulled back
```

Rules the code enforces (plan 6.1): no .py file outside `src/` and `tests/`; no setting has a
default; one job writes one folder and nothing overwrites anything; no code picks its own output
path; results record the code hash and the data manifest hash that produced them.

Settings added after runs exist (from 2026-09-26). A settings class lists such fields in
`ADDED_FIELDS` with the value that does what the code did before the field existed. A settings
dict must hold every field, or exactly the fields of an earlier date (then the listed values are
used); anything in between is refused, so a new sweep file cannot forget a new field. The hash
leaves an added field out while it holds its earlier value (`affect.settings.canonical_plain`), so
every configuration that ran before keeps its hash and run IDs, whichever form it is written in,
and the jobs still pending in the queue stay valid. Added so far: speed perturbation and mixup
(augment), the normalization layer and activation (model cnn), the gender head and evaluation
segments (train).

## Commands

On the PC, from the Cowork session or any shell with Python 3.10 or newer, in this folder:

```
PYTHONPATH=src python -m unittest discover -s tests -t .
PYTHONPATH=src python -m affect.cli ingest --dataset iemocap --name <name> --source <folder> --max-seconds 150
PYTHONPATH=src python -m affect.cli assemble --dataset iemocap \
    --source IEMOCAP_full_release=<folder> --source collaborator_wav=<folder> \
    --source phase0_audit=<folder> --max-seconds 150
PYTHONPATH=src python -m affect.cli verify-data --dataset iemocap
PYTHONPATH=src python -m affect.cli data-summary --dataset iemocap
```

`ingest` and `assemble` stop at `--max-seconds` and continue where they stopped when run again;
repeat until they print `"phase": "finalized"`.

Sending to the node (PC, PowerShell), then installing there:

```
powershell -ExecutionPolicy Bypass -File "C:\Affective Computing\affect\scripts\push.ps1" -Data
ssh <user>@<node>                         # the login in node.local.txt
bash ~/affect_incoming/unpack.sh affect_code_<utc>.tar.gz
bash ~/affect_incoming/unpack.sh affect_data_iemocap_<utc>.tar
bash ~/affect/scripts/setup_env.sh        # once: builds ~/venvs/affect
bash ~/affect/scripts/check.sh            # tests, verify-data, env-check on 4 GPUs
```

Results back to the PC: `bash ~/affect/scripts/pack.sh <path under artifacts/>` on the node,
then `pull.ps1 <name it prints>` on the PC. Several folders go in one archive (one pull, one
password prompt): `cd ~/affect/artifacts && bash ~/affect/scripts/pack.sh runs/<a> runs/<b> sweeps/<a> sweeps/<b>`;
include each sweep's `sweeps/<id>` record, which `analyze` needs. A sweep pulled before it
finished (a filler stage still pending) gets the rest later with
`bash ~/affect/scripts/pack.sh --after <name of the earlier archive> runs/<id>`: only files that
archive did not hold are packed (pull.ps1 refuses to overwrite), and a file it holds with other
content stops the pack. The earlier archive must still be in `~/affect_outgoing`. A sweep pulled in
more than two parts (the LIF screens, which run over several nights) needs every earlier archive of
it: `--after` may be given once per archive, `pack.sh --after <first> --after <second> runs/<id>`.

Checking that two copies of the code are the same (the "tree digest" quoted in HANDOFF.md and
VERIFICATION_LOG.md): run in the copy's root folder

```
find . \( -path ./data -o -path ./artifacts -o -path ./queue -o -path ./logs -o -path ./CODE_VERSIONS.log -o -name __pycache__ \) -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | cut -c1-16,65- | sha256sum | cut -c1-64
```

`LC_ALL=C` matters: the node's shell sorts file names in English order, which gives a different
digest for identical files (VERIFICATION_LOG entry 15). In `~/affect_next` also prune
`-path ./artifacts_preflight`, the 2026-09-26 pre-flight's own results (entry 22). CODE_VERSIONS.log exists only on the node
(unpack.sh writes it) and is left out.

## Running a sweep on the node

A sweep is a file in `sweeps/` (JSON): the complete base configuration, the folds, and stages
of configurations to try (plan 5.2). Every command below runs on the node, from any folder.

```
CUDA_VISIBLE_DEVICES=0 bash ~/affect/scripts/affect.sh smoke --sweep-file sweeps/cnn_screen_2.json --expected-gpus 1
bash ~/affect/scripts/affect.sh enqueue --sweep-file sweeps/cnn_confirm_1.json
bash ~/affect/scripts/affect.sh enqueue --sweep-file sweeps/cnn_screen_2.json
tmux new -d -s gpu0 "bash ~/affect/scripts/worker.sh 0 3"
tmux new -d -s gpu1 "bash ~/affect/scripts/worker.sh 1 3"
tmux new -d -s gpu2 "bash ~/affect/scripts/worker.sh 2 3"
tmux new -d -s gpu3 "bash ~/affect/scripts/worker.sh 3 3"
bash ~/affect/scripts/affect.sh status --sweep cnn_confirm_1 --top 20
bash ~/affect/scripts/affect.sh analyze --sweep cnn_confirm_1 --confidence 0.95 --n-boot 2000 --seed 20260926 --top 20
bash ~/affect/scripts/affect.sh select --sweep cnn_confirm_1 --config <first 12 characters of the chosen hash> --name cnn_final_1 --reason "<one sentence>"
bash ~/affect/scripts/affect.sh test-report --selection cnn_final_1 --confidence 0.95
```

From night 7 the workers start at night with the 10 am stop timer (see "Nights only" below): the line
is `artifacts/analyses/night7_prep_20260930/lines/8_evening_start.txt`, which also puts back any job a
dead worker left in `queue/running`.

`compare-runs` checks that the code reproduces earlier runs:
`bash ~/affect/scripts/affect.sh compare-runs --sweep cnn_regression_1 --differences-shown 3`
compares every finished run of the sweep with every folder of the same run ID in the other sweeps
(same configuration, fold and seed): validation predictions byte for byte, validation metrics
exactly without the per-epoch wall-clock times, and the two test files by SHA-256 only, never
parsed. It ends SAME (exit 0; every run identical to every copy, each with a copy made by other
code), DIFFERENT (exit 1) or INCOMPLETE (exit 3: runs still to finish, run it again later), and
saves `artifacts/checks/compare_runs_<sweep>_<utc>/` unless the verdict is INCOMPLETE. A sweep that
repeats only some earlier runs ends DIFFERENT by design: cnn_confirm_3's base repeats cnn_confirm_2's
25 runs of the same configuration, and its 19 candidates' 475 runs are new. There the table's rows are
what count: line 6 of `artifacts/analyses/night7_prep_20260930/lines` prints how many runs have a copy,
how many of those are the base's, and how many are identical to a copy made by other code, and
`base_repeat_check.py` in `night7_prep_20260930` checks the saved report run by run. Such a sweep is
DIFFERENT, and so saved, as soon as one new run exists: the report to check is the one made after all
of its runs finished (the check fails on any other).

`select` records the chosen configuration in `artifacts/selections/<name>/` and refuses unless a
finished `analyze` report of the sweep exists; a selection is never overwritten. `test-report`
then reads the test files of that configuration's runs only (test UA, WA, macro-F1 with t
intervals over seeds, per-class recall, UA by gender and kind) and saves
`artifacts/test_reports/<name>_<utc>/`. Nothing else in the code reads a test file's values
(`compare-runs` only hashes them).

Data protocols. The data section of a configuration names the label set (`four_merged`,
`four_without_excited`, `five`, `six`, `seven`), the subset (`all`, `impro`, `script`) and the split;
`src/affect/data/iemocap_splits.py` lists the 21 splits and what each holds out (twelve for
night 2, nine more from the literature review for night 4: random validation from the
training sessions, test session as validation (a labeled measurement, never a result), Session 5
as test, a quarter and three quarters of the training data, a script held out and its control,
within-speaker folds, and speaker enrollment). A sweep holds
one data protocol and lists exactly that split's folds (enqueue refuses anything else), so every
comparison inside a sweep is on the same validation sets. The baseline under every other protocol
is in `sweeps/cnn_protocol_*.json`, `sweeps/cnn_classes_*.json` and `sweeps/cnn_subset_*.json`
(600 jobs, priority 90). To smoke-test and enqueue all of them, stopping at the first problem:

```
cd ~/affect && for f in sweeps/cnn_protocol_*.json sweeps/cnn_classes_*.json sweeps/cnn_subset_*.json; do CUDA_VISIBLE_DEVICES=0 bash ~/affect/scripts/affect.sh smoke --sweep-file "$f" --expected-gpus 1 && bash ~/affect/scripts/affect.sh enqueue --sweep-file "$f" || { echo "STOPPED AT $f"; break; }; done
```

`analyze`, once a sweep has finished, compares every configuration with the same sweep's baseline
on the seeds they share: t intervals over seeds, a paired bootstrap that resamples seeds and
validation utterances, P(better), and Holm's correction across the comparisons. It reads
validation files only and saves `artifacts/analysis/<sweep_id>_<utc>/` (report.json, report.txt,
provenance.json). It compares within one sweep, and it needs scipy, so it runs on the node or in
the Cowork container, not in the PC's Python.

Night 1 (2026-09-25) ran `cnn_screen_1`, `cnn_screen_1_seeds` and `cnn_screen_1_random2`, 1,755
jobs. Night 2 (2026-09-25, 7:17 pm, to 10:26 am the next day) ran, highest priority first: `cnn_confirm_1`, the
baseline under every new data protocol, `cnn_grid_1`, the planned part of `cnn_screen_2`, the two
tuning-transfer sweeps, and the 200 extra configurations of `cnn_screen_2` as filler (HANDOFF.md
has the numbers). Screening uses 3 seeds per configuration from night 2 on. Night 3 (2026-09-26, 1:33 pm, to 3:40 pm the next day)
ran `cnn_confirm_2` on new seeds, the two confirmed finalists under every protocol (`cnn_tuned_*`),
`cnn_grid_2` and `cnn_screen_3`; HANDOFF.md has the numbers. Night 4 needs the night-4 code (written 2026-09-26 while night 3 ran):
`cnn_regression_1` first (configurations that ran before, which must give identical results), the
normalization-layer and activation study, augmentation and evaluation segments, the gender head,
the nine literature splits, more mel bands, and the first CNN + Transformer screen; the classical
sweeps run beside it on CPUs. Night 5 needs no new code, only sweep files: `cnn_transformer_screen_2`
(priority 100), `cnn_norm_final_2` (99), and the three night-3 finalists under every other split,
class set and subset (`cnn_tuned2_*`, priorities 98 to 72, one priority per sweep so that each
finishes before the next starts; the script-held-out pair lists only its 14 folds whose test set
holds a sad utterance). The 2,896 extras queued since nights 3 and 4 (priority 20) come after them.

The queue is meant to hold more than a night's work, most important first: workers always take
the highest-priority pending job, so a sweep enqueued later at a higher priority runs next without
stopping anything, as long as it needs no new code. In the morning, either let the rest run or
`touch ~/affect/queue/STOP` (from night 7 the stop timer does this at 10 am): each worker finishes
its current job and exits, and the pending jobs wait. Before pushing new code, stop every worker that way and wait until `tmux ls` lists no gpu
session (a running worker keeps the code it started with). Then `rm ~/affect/queue/STOP` before
starting workers again, because a worker exits at once while that file exists.

Nights only (from night 7, 2026-09-30, at Harrison's request): the line that starts the workers also
starts a timer, the tmux session `stopat10`, which sleeps until 10 am Pacific, touches
`~/affect/queue/STOP` and closes itself. The workers finish their current jobs and exit; the pending
jobs wait for the next evening, when the same line removes STOP, starts the workers and sets the next
timer (`artifacts/analyses/night7_prep_20260930/lines/8_evening_start.txt`). The timer is a
one-shot sleep, not a daemon: `tmux kill-session -t stopat10` cancels it. When the queue empties
before 10 am the workers exit on their own; the morning check (line 6) then sets STOP and cancels the
timer itself.

`smoke` tries every configuration of the sweep for one training step, runs a short real job (also
one for the first global- and the first speaker-normalized configuration, whose statistics a real
job computes itself), and times the base configuration; it must end with SMOKE OK before anything is enqueued.
`worker.sh <gpu> <processes>` pins its processes to one card with CUDA_VISIBLE_DEVICES; each
takes jobs until the queue is empty, and every job ends with a line starting JOB DONE or
JOB FAILED in `logs/worker_gpu<g>_<n>_<utc>.log`. `touch ~/affect/queue/STOP` makes every worker
exit after its current job. A worker also stops by itself after 5 failed jobs in a row, so a
problem that fails every job leaves the rest of the queue pending; once it is fixed,
`bash ~/affect/scripts/affect.sh requeue-failed --sweep cnn_screen_1` puts the failed jobs back.
Each job writes `artifacts/runs/<sweep_id>/<run_id>/`: spec, validation and test metrics, per-utterance predictions, timing, provenance and its log. `status` reads only
validation results.

Classical baselines (from 2026-09-26; `src/affect/classical/`). openSMILE functionals with an
SVM or a TPOT search run on CPUs in their own environment, `~/venvs/classical`, so the training
environment never changes. Once: `bash ~/affect/scripts/setup_classical_env.sh`, then
`bash ~/affect/scripts/classical.sh classical-selftest --dataset iemocap --label-set four_merged --subset all --n-jobs 4`
(must end with CLASSICAL SELFTEST OK). Then, once per feature set,
`bash ~/affect/scripts/classical.sh extract-functionals --dataset iemocap --label-set seven --subset all --feature-set eGeMAPSv02 --workers 16`
(into `artifacts/features/`), and a classical sweep file (`sweeps/classical_*.json`) runs in its own
tmux session with `bash ~/affect/scripts/classical.sh classical --sweep-file sweeps/classical_svm_1.json --max-consecutive-failures 3`.
It records the sweep and writes run folders like the GPU jobs, so `status`, `analyze`, `select` and
`test-report` read it; running the same file again resumes it. `enqueue` and `smoke` refuse
classical files.

Nothing in this project uses `/scratch` on the node. The environment is `~/venvs/affect`, model
weights go to `~/cache/huggingface`, and the data to `~/affect/data/iemocap/assembled`.
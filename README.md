# Example: the baseline CNN, from the config to the result

These are the real files behind one result of the project: the baseline CNN on IEMOCAP's four standard classes (neutral, happy+excited, angry, sad), session out, seeds 1 to 5. Its selection record is `cnn_base_session_out`, and its test UA is 0.559. The folders are numbered in the order the files were made, and each step links to the code that made them, at commit [`d193f4e`](https://github.com/basinhopping/affect/tree/d193f4ed296a20d68477549b91f8fe04af4fc78a).

The files are copied unchanged from the run, with two exceptions for this public repository. The run folder's `predictions_val.csv` and `predictions_test.csv` are left out, because they list IEMOCAP's labels utterance by utterance, and the corpus license keeps labels out of the repository. In `provenance.json`, the server's host name and my login in its paths are removed.

Two terms used throughout:

- **Session out:** IEMOCAP has five sessions, each with two speakers. Fold k tests session k, validates on session k + 1 (session 5 on session 1) and trains on the other three, so the test speakers are never in training.
- **UA (unweighted accuracy):** the recall of each class, averaged with equal weight.

## The pipeline

| Step | What happens | Files here | Code |
|---|---|---|---|
| 1. Config | I write one sweep file that holds every setting | [`1_config/`](1_config) | [`spec.py`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L43-L80) |
| 2. Enqueue | the sweep becomes one job per configuration, fold and seed | [`2_job/`](2_job) | [`design.py`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L78-L224) |
| 3. Run | a GPU worker trains one job and writes its run folder | [`3_run/`](3_run/cf0495a296e9_test_S1_seed1) | [`loop.py`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L308-L501) |
| 4. Analyze | the configurations are compared on validation files only | [`4_analyze/`](4_analyze/report.txt) | [`validation.py`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L189-L218) |
| 5. Select | the choice is written down as a record | [`5_select/`](5_select/selection.json) | [`selection.py`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L38-L61) |
| 6. Test report | only now are test files read, and only those of the recorded runs | [`6_test_report/`](6_test_report/report.txt) | [`selection.py`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L69-L110) |

## 1. The config I send up: [`1_config/`](1_config)

[`cnn_confirm_1_sweep_file.json`](1_config/cnn_confirm_1_sweep_file.json) is the sweep that produced the record (night 2). It is the same as [`sweeps/cnn_confirm_1.json`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_confirm_1.json). [`base_cnn_sweep.json`](1_config/base_cnn_sweep.json) is the baseline on its own, with seed 1 only. A sweep file has three parts:

- `folds`: the five session-out folds, `test_S1` to `test_S5`.
- `base` ([lines 11 to 68](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_confirm_1.json#L11-L68)): every setting of the baseline, with nothing left to a default.
  - `data`: IEMOCAP, the four merged classes, all 5,531 utterances, one session held out per fold.
  - `features`: log-mel spectrograms, 25 ms windows every 10 ms, 64 mel bands from 0 to 8,000 Hz, normalized per utterance.
  - `augment`: all zero, so the baseline has no SpecAugment.
  - `model`: the CNN, with four blocks of 32, 64, 128 and 256 channels, 3x3 kernels, 2x2 pooling, mean pooling over time and dropout 0.3.
  - `train`: up to 60 epochs, batch size 32, learning rate 0.001 (2 warmup epochs, then a cosine decay) and weight decay 0.0001. Training stops after 10 epochs without a better validation UA, and each epoch uses random 5 s crops.
- `stages` ([from line 69](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/sweeps/cnn_confirm_1.json#L69)): what to run. `baseline` runs the base on seeds 1 to 5. `candidates` holds the night's 16 other configurations, each the base with a few settings changed.

When the code reads a job, [`validate_job`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L43-L55) checks every section against its settings class with [`validate`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/settings.py#L217-L242). A missing or unknown setting stops it with an error, so a typo can never fall back to a default. The one allowance is for settings added to the code later: a file written before such a setting existed gets the value the code had then. For the CNN, those are [batch norm and ReLU](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L46-L47).

[`config_hash`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/runs/spec.py#L79-L80) then takes the SHA-256 of the checked settings. Its first 12 characters, `cf0495a296e9`, name the baseline from here on: in the job, the run folder and the record. The same settings always give the same name.

## 2. One job: [`2_job/`](2_job)

On the node, a smoke test runs every configuration of the sweep for one step, and then `enqueue` queues the sweep:

```
CUDA_VISIBLE_DEVICES=0 bash ~/affect/scripts/affect.sh smoke --sweep-file sweeps/cnn_confirm_1.json --expected-gpus 1
bash ~/affect/scripts/affect.sh enqueue --sweep-file sweeps/cnn_confirm_1.json
```

[`enqueue`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/enqueue.py#L29-L60) calls [`expand`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/design.py#L78-L224), which turns the stages into jobs: 17 configurations × 5 folds × 5 seeds = 425 job files in `queue/pending/`. One of them is [`cf0495a296e9_test_S1_seed1.json`](2_job/cf0495a296e9_test_S1_seed1.json). It holds the full settings, the fold `test_S1`, seed 1, the settings hash and the run id.

Fold `test_S1` tests on session 1 (1,085 utterances), validates on session 2 (1,023 utterances) and trains on sessions 3, 4 and 5.

## 3. What one job does: [`3_run/`](3_run/cf0495a296e9_test_S1_seed1)

Workers run in tmux, three to a GPU (`tmux new -d -s gpu0 "bash ~/affect/scripts/worker.sh 0 3"`). Each [worker](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/worker.py#L71-L107) claims the next job by renaming its file into `running/` ([`take`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/sweep/queue.py#L39-L47)). A rename succeeds for only one worker, so no job runs twice. The worker then calls [`train_job`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L308-L324), which loads the audio and the fold's training, validation and test utterances.

### 3a. From audio to the spectrogram

Each training batch goes through these five steps ([`Step.loss`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L179-L185)):

1. **Crop** ([`crop_arrays`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L121-L136)). Each epoch takes a random 5 s crop of every training utterance: 80,000 samples at 16 kHz. A shorter utterance is zero-padded, and its real length is kept so that the padding is ignored later. The 16-bit samples are scaled to the range from −1 to 1.
2. **Short-time Fourier transform** ([`LogMel.__call__`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L113-L118)). A Hann window of 25 ms (400 samples) moves in steps of 10 ms (160 samples), with a 512-point FFT and centred frames. A 5 s crop gives 1 + 80,000 / 160 = 501 frames, and the power of each frame has 257 frequency bins.
3. **Mel filterbank** ([built here](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L55-L67), [applied here](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L119)). 64 triangular filters between 0 and 8,000 Hz sum the power into 64 bands. They are spaced on the mel scale, which is linear below 1,000 Hz and logarithmic above it, like human pitch perception. The weights are those of librosa's default filterbank, and a test compares the two.
4. **Log** ([line 120](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/features/logmel.py#L120)). The log of (mel power + 0.000001), so that silence stays finite.
5. **Per-utterance normalization** ([`Normalizer`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L68-L72)). Each mel band is scaled to mean 0 and SD 1 over the utterance's real frames, and the padded frames are set to 0 ([line 82](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L82)).

A batch comes out as 32 log-mel spectrograms of 64 bands × 501 frames. Validation and test use whole utterances instead of crops, padded to the longest utterance in each batch ([`eval_batches`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/data.py#L155-L166)).

### 3b. Inside the CNN

For one 5 s crop:

| Layer | Output (channels × bands × frames) | Parameters |
|---|---|---|
| Input: the log-mel spectrogram | 1 × 64 × 501 | |
| Block 1: 3x3 convolution from 1 to 32 channels, batch norm, ReLU, 2x2 max pooling | 32 × 32 × 250 | 288 + 64 |
| Block 2: the same, 32 to 64 channels | 64 × 16 × 125 | 18,432 + 128 |
| Block 3: the same, 64 to 128 channels | 128 × 8 × 62 | 73,728 + 256 |
| Block 4: the same, 128 to 256 channels | 256 × 4 × 31 | 294,912 + 512 |
| Mean over the 4 frequency rows | 256 × 31 | |
| Mean over the real frames (mean pooling) | 256 | |
| Dropout 0.3, then a linear layer to the 4 class scores | 4 | 1,028 |
| Total | | 389,348 |

In each block, the first number is the convolution's weights and the second is batch norm's scale and shift.

- [`conv_front_end`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L165-L187) builds the four blocks. Each convolution pads its input by one, so it keeps the size, and it has no bias because batch norm follows it. Each 2x2 max pooling halves both frequency and time.
- [`embed`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L235-L242) averages over frequency, works out which time steps are real rather than padding ([`frames_after`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L203-L210)), and pools over time with [`masked_pool`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L98-L121).
  - Mean pooling ([lines 102 to 106](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L102-L106)) keeps each channel's mean over time.
  - SD pooling, `stats` in the settings ([lines 113 to 115](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L113-L115)), keeps the mean and the standard deviation side by side, so 256 channels give 512 numbers.
- [`classifier_head`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/models/cnn.py#L190-L200) applies dropout and the output layer. The loss is the cross-entropy of the 4 scores ([`soft_cross_entropy`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L138-L144)).

### 3c. Training

- **Optimizer** ([lines 351 to 363](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L351-L363)): AdamW with weight decay 0.0001. The learning rate rises over the first 2 epochs to 0.001, then follows a cosine decay ([`lr_factor`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L40-L46)).
- **Each epoch** ([lines 383 to 445](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L383-L445)): batches of 32 crops, each with a forward pass, the loss, a backward pass and an optimizer step ([lines 392 to 402](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L392-L402)). Then validation UA is computed on whole utterances ([lines 411 to 412](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L411-L412)).
- **Keep the best epoch** ([lines 433 to 436](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L433-L436)): when validation UA beats the best so far, the weights are copied.
- **Stop** ([lines 443 to 445](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L443-L445)): after 10 epochs without a higher validation UA, or at epoch 60.
- **At the end** ([lines 448 to 501](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/train/loop.py#L448-L501)): the kept weights predict validation and test once, and the run folder's files are returned.

This run kept epoch 6 (validation UA 0.624) and stopped after epoch 16. It trained for 32.6 s on an NVIDIA H100 NVL, with a peak of 918 MiB of GPU memory. From [`metrics_val.json`](3_run/cf0495a296e9_test_S1_seed1/metrics_val.json):

| Epoch | Training loss | Validation UA |
|---|---|---|
| 1 | 1.254 | 0.304 |
| 2 | 1.159 | 0.376 |
| 3 | 1.099 | 0.471 |
| 4 | 1.079 | 0.544 |
| 5 | 1.038 | 0.399 |
| 6 | 1.047 | 0.624 (kept) |
| 7 | 1.011 | 0.419 |
| 8 | 1.002 | 0.544 |
| 9 | 0.983 | 0.431 |
| 10 | 0.983 | 0.493 |
| 11 | 0.967 | 0.594 |
| 12 | 0.974 | 0.442 |
| 13 | 0.943 | 0.603 |
| 14 | 0.945 | 0.449 |
| 15 | 0.929 | 0.555 |
| 16 | 0.919 | 0.557 |

### The run folder: [`3_run/cf0495a296e9_test_S1_seed1/`](3_run/cf0495a296e9_test_S1_seed1)

- [`spec.json`](3_run/cf0495a296e9_test_S1_seed1/spec.json): the job from step 2.
- [`metrics_val.json`](3_run/cf0495a296e9_test_S1_seed1/metrics_val.json): the curve above, the kept epoch, and the validation report at the kept epoch. The report has recall, precision, F1 and the confusion matrix, by class and by kind (improvised or scripted).
- [`metrics_test.json`](3_run/cf0495a296e9_test_S1_seed1/metrics_test.json): the same report for test. Nothing reads it until step 6.
- [`efficiency.json`](3_run/cf0495a296e9_test_S1_seed1/efficiency.json): parameters, seconds, epochs and peak GPU memory.
- [`provenance.json`](3_run/cf0495a296e9_test_S1_seed1/provenance.json): the hashes of the code, the data and the folds, the package versions and the GPU.
- `predictions_val.csv` and `predictions_test.csv` (not in this repository; see the top): one row per utterance, with its id, speaker, kind, label, the prediction and the 4 class probabilities.

## 4. Compare on validation: [`4_analyze/report.txt`](4_analyze/report.txt)

```
bash ~/affect/scripts/affect.sh analyze --sweep cnn_confirm_1 --confidence 0.95 --n-boot 2000 --seed 20260926 --top 20
```

[`analyze`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/validation.py#L189-L218) reads only the validation files of the sweep's 425 runs. Each seed's UA is the mean of its five folds. The baseline line reads validation UA 0.5887 [0.5812, 0.5961] over seeds 1 to 5.

Every other configuration is compared with the baseline:

- **The difference in UA.**
- **A paired bootstrap interval** ([`paired_bootstrap_ua_difference`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L60-L114)). Each of the 2,000 draws resamples every fold's utterances, using the same draw for both configurations, which is what makes it paired. It also resamples each configuration's seeds.
- **A Holm-adjusted p** ([`holm`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/eval/stats.py#L138-L148)) over all the comparisons, so that one lucky configuration does not count as a finding.

## 5. Record the choice: [`5_select/selection.json`](5_select/selection.json)

```
bash ~/affect/scripts/affect.sh select --sweep cnn_confirm_1 --config cf0495a296e9 --name cnn_base_session_out --reason "The baseline, fixed before the run; the session-out reference row for the protocol spread and the gender comparison."
```

[`selection_record`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L38-L61) writes the record: the settings, the validation numbers, the reason, and the analyze report it came from, with that report's SHA-256. A record is never overwritten, and `test-report` refuses to run without one.

## 6. The test result: [`6_test_report/report.txt`](6_test_report/report.txt)

```
bash ~/affect/scripts/affect.sh test-report --selection cnn_base_session_out --confidence 0.95
```

[`test_report`](https://github.com/basinhopping/affect/blob/d193f4ed296a20d68477549b91f8fe04af4fc78a/src/affect/analysis/selection.py#L69-L110) reads the test files of the record's 25 runs (5 folds × 5 seeds) and nothing else:

- test UA 0.5590 [0.5494, 0.5685] (95% t interval over seeds), WA 0.5295 and macro-F1 0.5193;
- recall by class: neutral 0.423, happy+excited 0.378, angry 0.692 and sad 0.743;
- every seed and fold. The run in step 3 is seed 1 on `test_S1`, with test UA 0.5941.

This is the baseline's 0.559 in the meeting results.

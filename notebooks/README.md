# Notebooks

`cnn_3block.ipynb` shows how one experiment is written and run, with the 3-block CNN as the example: a SETTINGS
cell with every setting on its own line, a check that validates and hashes them, the sweep that varies them
(expanded into one job per configuration, fold and seed), and the results read from the run folders: one run's
training curve, the screen and its confirmation on validation, and the recorded configuration's test result.

Run it from this folder with Python 3.10 or newer, pandas and matplotlib. The settings, check and sweep cells need
only the repository; the result cells read `../artifacts/`, which is not in the repository, and say so when it is
missing. A new experiment is written the same way: copy the SETTINGS cell, change what the experiment changes,
check it, and save the SWEEP as `sweeps/<sweep_id>.json`, which the node's scripts smoke-test and run.

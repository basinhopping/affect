"""Classical baselines (added 2026-09-26): openSMILE functionals per utterance with an SVM, or with a
TPOT search over scikit-learn pipelines (Dr. Luitel asked about TPOT; it cannot tune the neural
models, so it runs here, on fixed features, as an AutoML baseline).

    settings.py      the settings of a classical job: data, functionals, method, protocol
    functionals.py   extract openSMILE functionals once per dataset and feature set; load them
    methods.py       the methods, registered in affect.registry.CLASSICAL: svm, tpot
    runner.py        expand a classical sweep file, record it, run its jobs into run folders

A classical job uses the same folds as the neural models (the dataset's split schemes), picks its
settings on the validation set only, and writes the same run-folder files (spec.json,
metrics_val.json, predictions_val.csv, metrics_test.json, predictions_test.csv, ...), so analyze,
select and test-report work on it unchanged. It needs scikit-learn, opensmile and tpot, which live
in their own environment on the node (scripts/setup_classical_env.sh), not in the training one.
"""

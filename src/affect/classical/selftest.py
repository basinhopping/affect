"""classical-selftest: run once after building the classical environment, before any classical sweep.

1. openSMILE on three real utterances of the assembled copy: every feature set gives its known
   number of functionals, all finite.
2. The SVM on a synthetic four-class problem with known clusters, through the same method code a
   real job runs: validation UA above 0.9 for both kernels.
3. TPOT on the same problem with a tiny search (2 generations of 6 pipelines, the light operators):
   it must run, return scores, and beat chance.
Ends with CLASSICAL SELFTEST OK or CLASSICAL SELFTEST FAILED.
"""
import traceback

import numpy as np

from affect import paths
from affect.classical.functionals import EXPECTED_WIDTH, functionals_of
from affect.classical.methods import unweighted_accuracy
from affect.classical.settings import SVMSettings, TPOTSettings
from affect.registry import CLASSICAL, DATASETS
from affect.settings import validate
from affect.train.data import read_int16


def _synthetic(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, size=(4, 20))

    def draw(n):
        y = np.arange(n) % 4
        return centres[y] + rng.normal(0, 1, size=(n, 20)), y

    (xt, yt), (xv, yv), (xs, ys) = draw(400), draw(120), draw(120)
    return {"x_train": xt, "y_train": yt, "x_val": xv, "y_val": yv, "x_test": xs, "y_test": ys, "n_classes": 4}


def selftest(dataset: str, data_settings: dict, n_jobs: int) -> bool:
    ok = True
    try:
        import opensmile

        entry = DATASETS.get(dataset)
        utterances = entry.target().utterances(validate(entry.settings_class, data_settings, "data"))[:3]
        root = paths.assembled_dir(dataset)
        for feature_set, width in EXPECTED_WIDTH.items():
            for u in utterances:
                names, vector = functionals_of(read_int16(root / u.audio_path, u.n_frames), u.sample_rate, feature_set)
                if vector.shape != (width,) or len(names) != width or not np.isfinite(vector).all():
                    raise ValueError(f"{feature_set} on {u.utt_id}: {vector.shape} values, {len(names)} names, "
                                     f"{int((~np.isfinite(vector)).sum())} not finite")
            print(f"ok  openSMILE {getattr(opensmile, '__version__', '?')} {feature_set}: {width} functionals "
                  f"on {len(utterances)} utterances")
    except Exception:
        ok = False
        print("FAILED  openSMILE")
        traceback.print_exc()
    data = _synthetic(0)
    for kernel, gammas in (("linear", ()), ("rbf", (0.5, 1.0))):
        try:
            s = SVMSettings(kernel=kernel, c_values=(0.1, 1.0), gamma_scales=gammas, class_weight="balanced")
            out = CLASSICAL.get("svm").target(s, data, 1, "train", lambda m: None)
            val = unweighted_accuracy(data["y_val"], out["val_scores"].argmax(axis=1), 4)
            test = unweighted_accuracy(data["y_test"], out["test_scores"].argmax(axis=1), 4)
            if val < 0.9 or test < 0.9 or not np.allclose(out["val_scores"].sum(axis=1), 1.0):
                raise ValueError(f"validation UA {val:.3f}, test UA {test:.3f}")
            print(f"ok  svm {kernel}: validation UA {val:.3f}, test UA {test:.3f}, chose {out['chosen']}")
        except Exception:
            ok = False
            print(f"FAILED  svm {kernel}")
            traceback.print_exc()
    try:
        s = TPOTSettings(generations=2, population_size=6, offspring_size=6, mutation_rate=0.9, crossover_rate=0.1,
                         operators="light", max_time_mins=5.0, max_eval_time_mins=1.0, n_jobs=n_jobs)
        out = CLASSICAL.get("tpot").target(s, data, 1, "train", print)
        test = unweighted_accuracy(data["y_test"], out["test_scores"].argmax(axis=1), 4)
        if test < 0.5:
            raise ValueError(f"test UA {test:.3f}")
        print(f"ok  tpot: test UA {test:.3f}, {out['chosen']['pipelines_evaluated']} pipelines evaluated, "
              f"chose {out['chosen']['pipeline']}")
    except Exception:
        ok = False
        print("FAILED  tpot")
        traceback.print_exc()
    print("CLASSICAL SELFTEST OK" if ok else "CLASSICAL SELFTEST FAILED")
    return ok

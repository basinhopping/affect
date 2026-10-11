"""openSMILE functionals, extracted once per dataset and feature set (added 2026-09-26).

extract() reads every utterance of the chosen label set from the assembled copy (the same int16
samples the neural models read, affect.train.data.read_int16), runs openSMILE's functionals on it
(opensmile.Smile(...).process_signal, 16 kHz), and writes

    artifacts/features/<dataset>/<feature_set>/values.npy   float32 [utterances, features]
    artifacts/features/<dataset>/<feature_set>/utt_ids.txt  one utterance ID per row of values.npy
    artifacts/features/<dataset>/<feature_set>/names.txt    one feature name per column
    artifacts/features/<dataset>/<feature_set>/record.json  what was extracted, from what, with what

The folder is written once (affect.runs.store): extracting the same feature set again refuses.
Take the largest label set when extracting (IEMOCAP: seven), so every label set's utterances are
there; load() refuses a job whose utterances are missing. opensmile is imported only here, so the
rest of the project runs without it.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from affect import paths
from affect.hashing import sha256_file, sha256_text
from affect.registry import DATASETS
from affect.runs.provenance import collect
from affect.runs.store import OutputFolder
from affect.settings import validate
from affect.train.data import read_int16

EXPECTED_WIDTH = {"eGeMAPSv02": 88, "ComParE_2016": 6373, "emobase": 988}
_SMILE: dict = {}          # one openSMILE extractor per feature set in each worker process


def features_dir(dataset: str, feature_set: str) -> Path:
    return paths.artifacts_dir() / "features" / dataset / feature_set


def _smile(feature_set: str):
    import opensmile

    if feature_set not in _SMILE:
        _SMILE[feature_set] = opensmile.Smile(feature_set=getattr(opensmile.FeatureSet, feature_set),
                                              feature_level=opensmile.FeatureLevel.Functionals)
    return _SMILE[feature_set]


def functionals_of(samples: np.ndarray, sample_rate: int, feature_set: str) -> tuple:
    """(feature names, float32 vector) of one int16 waveform."""
    frame = _smile(feature_set).process_signal(samples.astype(np.float32) / 32768.0, sample_rate)
    if frame.shape[0] != 1:
        raise ValueError(f"openSMILE gave {frame.shape[0]} rows of functionals for one utterance")
    return [str(c) for c in frame.columns], frame.to_numpy(dtype=np.float32)[0]


def _chunk(args) -> tuple:
    root, items, sample_rate, feature_set = args
    names, rows = None, []
    for audio_path, n_frames in items:
        these, vector = functionals_of(read_int16(Path(root) / audio_path, n_frames), sample_rate, feature_set)
        if names is None:
            names = these
        elif these != names:
            raise ValueError(f"{audio_path}: openSMILE returned different feature names")
        rows.append(vector)
    return names, np.stack(rows)


def extract(dataset: str, data_settings: dict, feature_set: str, workers: int, log) -> Path:
    """Extract the functionals of every utterance of these data settings into features_dir()."""
    if feature_set not in EXPECTED_WIDTH:
        raise ValueError(f"unknown feature set {feature_set!r}; known: {sorted(EXPECTED_WIDTH)}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    entry = DATASETS.get(dataset)
    settings = validate(entry.settings_class, data_settings, "data")
    ds = entry.target()
    utterances = ds.utterances(settings)
    root = paths.assembled_dir(dataset)
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates {rates}")
    rate = rates.pop()
    folder = OutputFolder(features_dir(dataset, feature_set).parent, feature_set).begin()
    try:
        start = time.perf_counter()
        items = [(u.audio_path, u.n_frames) for u in utterances]
        size = max(1, min(64, len(items) // (4 * workers) or 1))
        chunks = [(str(root), items[k:k + size], rate, feature_set) for k in range(0, len(items), size)]
        names, parts = None, []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, (these, block) in enumerate(pool.map(_chunk, chunks), 1):
                if names is None:
                    names = these
                elif these != names:
                    raise ValueError("feature names differ between worker processes")
                parts.append(block)
                if n % 20 == 0 or n == len(chunks):
                    log(f"  {sum(p.shape[0] for p in parts)} of {len(items)} utterances, "
                        f"{time.perf_counter() - start:.0f}s")
        values = np.concatenate(parts).astype(np.float32)
        if values.shape != (len(utterances), EXPECTED_WIDTH[feature_set]):
            raise ValueError(f"{feature_set}: got {values.shape}, expected ({len(utterances)}, "
                             f"{EXPECTED_WIDTH[feature_set]})")
        np.save(folder.path("values.npy"), values)
        folder.write_text("utt_ids.txt", "".join(f"{u.utt_id}\n" for u in utterances))
        folder.write_text("names.txt", "".join(f"{name}\n" for name in names))
        import opensmile

        _, manifest_sha = sha256_file(root / "MANIFEST.csv")
        record = {"dataset": dataset, "data_settings": data_settings, "feature_set": feature_set,
                  "feature_level": "Functionals", "n_utterances": int(values.shape[0]),
                  "n_features": int(values.shape[1]), "names_sha256": sha256_text("\n".join(names)),
                  "non_finite_values": int((~np.isfinite(values)).sum()), "sample_rate": rate,
                  "opensmile_version": getattr(opensmile, "__version__", None),
                  "data_manifest_sha256": manifest_sha, "seconds": round(time.perf_counter() - start, 1),
                  "workers": workers}
        folder.write_json("record.json", record)
        folder.write_json("provenance.json", collect("affect.cli extract-functionals"))
    except BaseException as err:
        folder.fail(err)
        raise
    done = folder.complete()
    log(f"saved {done}: {values.shape[0]} utterances x {values.shape[1]} features, "
        f"{record['non_finite_values']} non-finite values")
    return done


def load(dataset: str, feature_set: str, utt_ids: list) -> tuple:
    """(values float64 [len(utt_ids), features] in the order given, feature names, record)."""
    folder = features_dir(dataset, feature_set)
    if not (folder / "record.json").is_file():
        raise FileNotFoundError(f"no functionals at {folder}; run extract-functionals first")
    record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    ids = (folder / "utt_ids.txt").read_text(encoding="utf-8").split()
    names = (folder / "names.txt").read_text(encoding="utf-8").splitlines()
    values = np.load(folder / "values.npy")
    if values.shape != (len(ids), len(names)):
        raise ValueError(f"{folder}: values {values.shape} do not match {len(ids)} IDs and {len(names)} names")
    row = {u: i for i, u in enumerate(ids)}
    missing = [u for u in utt_ids if u not in row]
    if missing:
        raise ValueError(f"{len(missing)} utterances have no {feature_set} functionals, e.g. {missing[:3]}; "
                         "extract them with a label set that includes them")
    return values[[row[u] for u in utt_ids]].astype(np.float64), names, record

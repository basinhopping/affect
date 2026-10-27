"""Pretrained speech encoders' hidden layers as utterance vectors, for the classical pipeline (added 2026-10-19).

Dr. Luitel's second request of 2026-10-12: pretrained models on the four setups, from the audio alone (no transcripts).
A frozen self-supervised encoder reads each utterance once, and for every hidden layer the mean of that layer's frame
vectors over the whole utterance is the utterance's vector. Each layer becomes one feature set of the classical
pipeline, "<model>_l<k>" (affect.classical.settings.PRETRAINED_FEATURE_SETS), which affect.classical.functionals.load
reads like openSMILE's functionals; an SVM on it, its settings chosen on validation like every other model here, is a
probe of what that layer holds about emotion. The encoders (affect.classical.settings.PRETRAINED_MODELS) were
pretrained on unlabeled speech, and none of these checkpoints was fine-tuned on transcripts:

    wavlm_base_plus   microsoft/wavlm-base-plus    WavLM Base+ (Chen et al. 2022), 94,000 hours of unlabeled speech
    hubert_base       facebook/hubert-base-ls960   HuBERT Base (Hsu et al. 2021), LibriSpeech's 960 hours, unlabeled
    wav2vec2_base     facebook/wav2vec2-base       wav2vec 2.0 Base (Baevski et al. 2020), the same 960 hours
    wavlm_large       microsoft/wavlm-large        WavLM Large, the same 94,000 hours
    hubert_large      facebook/hubert-large-ll60k  HuBERT Large, Libri-Light's 60,000 hours, unlabeled
    wav2vec2_large    facebook/wav2vec2-large-lv60 wav2vec 2.0 Large, the same 60,000 hours

The Base models have 12 transformer layers of width 768 (13 hidden states), the Large ones 24 of width 1024 (25).

    download(model, cache_dir, log)      the model's config, feature-extractor config and weights from the Hugging
                                         Face hub into cache_dir (~/cache/huggingface), once; returns the snapshot
    extract(dataset, data_settings, model, snapshot, device, log, pooling="mean")
        writes artifacts/features/<dataset>/<model>_l<k>/ for k = 0 (the encoder's projected convolutional output) to
        the last transformer layer (12 or 24): values.npy float32 [utterances, width], utt_ids.txt, names.txt,
        record.json and provenance.json, each folder written once (affect.runs.store), so extracting again refuses.
        pooling "mean_sd" (added 2026-10-20) writes <model>_l<k>_msd instead: the layer's mean over its frames, then
        its standard deviation over them (population SD), side by side, [utterances, 2 x width].
    derive_layer_average(dataset, model, log)   (added 2026-10-20)
        writes artifacts/features/<dataset>/<model>_lavg/: the mean over every layer of that layer's mean vector,
        from the <model>_l<k> folders already extracted; no encoder is run.

Each utterance is the same int16 samples the neural models read (affect.train.data.read_int16), scaled to [-1, 1) and
passed alone (no padding) through the model's own feature extractor (its preprocessor_config.json: 16 kHz, and the
per-utterance normalisation the model was trained with) and the encoder in eval mode, float32, without gradients.
torch and transformers are imported only in load_encoder and download, so the rest of the project runs without them.
"""
import json
import time
from pathlib import Path

import numpy as np

from affect import paths
from affect.classical.functionals import features_dir
from affect.classical.settings import PRETRAINED_MODELS
from affect.hashing import sha256_file, sha256_text
from affect.registry import DATASETS
from affect.runs.provenance import collect
from affect.runs.store import OutputFolder
from affect.settings import validate
from affect.train.data import read_int16

REPOS = {"wavlm_base_plus": "microsoft/wavlm-base-plus", "hubert_base": "facebook/hubert-base-ls960",
         "wav2vec2_base": "facebook/wav2vec2-base", "wavlm_large": "microsoft/wavlm-large",
         "hubert_large": "facebook/hubert-large-ll60k", "wav2vec2_large": "facebook/wav2vec2-large-lv60"}
CONFIG_FILES = ("config.json", "preprocessor_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")     # the first the hub holds is downloaded


POOLINGS = {"mean": "", "mean_sd": "_msd"}        # pooling -> the suffix of its feature sets' names (2026-10-20)


def feature_sets(model: str, pooling: str = "mean") -> list:
    if model not in PRETRAINED_MODELS:
        raise ValueError(f"unknown pretrained model {model!r}; known: {sorted(PRETRAINED_MODELS)}")
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; known: {sorted(POOLINGS)}")
    return [f"{model}_l{k}{POOLINGS[pooling]}" for k in range(PRETRAINED_MODELS[model][0])]


def weights_of(snapshot: Path) -> Path:
    found = [snapshot / name for name in WEIGHT_FILES if (snapshot / name).is_file()]
    if not found:
        raise FileNotFoundError(f"no weights ({' or '.join(WEIGHT_FILES)}) in {snapshot}")
    return found[0]


def download(model: str, cache_dir: Path, log) -> Path:
    """The model's files into the Hugging Face cache; returns the snapshot folder (its name is the commit)."""
    feature_sets(model)
    from huggingface_hub import HfApi, snapshot_download

    repo = REPOS[model]
    api = HfApi()
    try:
        files = set(api.list_repo_files(repo))
    except AttributeError:          # a huggingface_hub without list_repo_files
        files = {s.rfilename for s in api.model_info(repo).siblings}
    missing = [name for name in CONFIG_FILES if name not in files]
    weights = [name for name in WEIGHT_FILES if name in files]
    if missing or not weights:
        raise FileNotFoundError(f"{repo} lacks {missing or 'weights'}; it has {sorted(files)[:20]}")
    snapshot = Path(snapshot_download(repo, cache_dir=str(cache_dir), allow_patterns=list(CONFIG_FILES) + weights[:1]))
    n_bytes, sha = sha256_file(weights_of(snapshot))
    log(f"{model}: {repo} at {snapshot} (commit {snapshot.name}), {weights_of(snapshot).name} {n_bytes:,} bytes, "
        f"SHA-256 {sha}")
    return snapshot


def pool_layers(layers: list, pooling: str) -> np.ndarray:
    """One utterance's hidden layers, each a torch tensor [frames, width], pooled over the frames: float32
    [layers, width], each layer's mean; with pooling mean_sd, [layers, 2 x width], each layer's mean and then its
    population SD, the square root of the mean squared deviation from that mean (written out, so that it does not
    depend on how a torch version names std's correction argument)."""
    import torch

    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; known: {sorted(POOLINGS)}")
    means = torch.stack([h.mean(dim=0) for h in layers])
    if pooling == "mean":
        return means.cpu().numpy().astype(np.float32)
    sds = torch.stack([(h - h.mean(dim=0)).pow(2).mean(dim=0).sqrt() for h in layers])
    return torch.cat([means, sds], dim=1).cpu().numpy().astype(np.float32)


def load_encoder(snapshot: Path, device: str, pooling: str = "mean"):
    """(encode, info): encode(int16 samples) -> float32 [layers, width], the mean of every hidden layer over time; with
    pooling mean_sd, [layers, 2 x width], each layer's mean and then its population SD over time."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    rate = int(extractor.sampling_rate)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).to(device).eval()

    def encode(samples: np.ndarray) -> np.ndarray:
        wave = samples.astype(np.float32) / 32768.0
        inputs = extractor(wave, sampling_rate=rate, return_tensors="pt")
        kwargs = {"input_values": inputs["input_values"].to(device)}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(**kwargs, output_hidden_states=True)
        layers = [h[0].float() for h in out.hidden_states]           # each [frames, width]
        if layers[0].shape[0] < 1:
            raise ValueError("the encoder gave no frame")
        return pool_layers(layers, pooling)

    config = model.config
    info = {"model_type": config.model_type, "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size, "layers": config.num_hidden_layers + 1,
            "feature_extractor": {k: getattr(extractor, k, None) for k in
                                  ("sampling_rate", "do_normalize", "return_attention_mask", "padding_value")},
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "device": str(device), "dtype": "float32", "pooling": pooling}
    return encode, info


def extract(dataset: str, data_settings: dict, model: str, snapshot: Path, device: str, log, encoder=None,
            pooling: str = "mean") -> list:
    """Every utterance of these data settings through the encoder; one feature folder per hidden layer."""
    names = feature_sets(model, pooling)
    n_layers, layer_width = PRETRAINED_MODELS[model]
    width = layer_width * (2 if pooling == "mean_sd" else 1)
    snapshot = Path(snapshot)
    existing = [n for n in names if features_dir(dataset, n).exists()]
    if existing:
        raise FileExistsError(f"feature sets exist already, nothing extracted: {existing}")
    entry = DATASETS.get(dataset)
    settings = validate(entry.settings_class, data_settings, "data")
    utterances = entry.target().utterances(settings)
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates {rates}")
    rate = rates.pop()
    root = paths.assembled_dir(dataset)
    weights = weights_of(snapshot)
    n_bytes, weights_sha = sha256_file(weights)
    encode, info = encoder if encoder is not None else load_encoder(snapshot, device, pooling)
    if info["feature_extractor"].get("sampling_rate") != rate:
        raise ValueError(f"{model} expects {info['feature_extractor'].get('sampling_rate')} Hz, the data is {rate} Hz")
    if info["layers"] != n_layers or info["hidden_size"] != layer_width:
        raise ValueError(f"{model}: the encoder has {info['layers']} layers of {info['hidden_size']}, the table says "
                         f"{n_layers} of {layer_width}")
    start = time.perf_counter()
    values = np.empty((n_layers, len(utterances), width), dtype=np.float32)
    frames = np.empty(len(utterances), dtype=np.int64)
    for i, u in enumerate(utterances):
        samples = read_int16(root / u.audio_path, u.n_frames)
        v = encode(samples)
        if v.shape != (n_layers, width):
            raise ValueError(f"{u.utt_id}: the encoder gave {v.shape}, expected {(n_layers, width)}")
        values[:, i] = v
        frames[i] = samples.shape[0]
        if (i + 1) % 500 == 0 or i + 1 == len(utterances):
            log(f"  {i + 1} of {len(utterances)} utterances, {time.perf_counter() - start:.0f}s")
    non_finite = int((~np.isfinite(values)).sum())
    if non_finite:
        raise ValueError(f"{non_finite} non-finite values: nothing written")
    _, manifest_sha = sha256_file(root / "MANIFEST.csv")
    written = []
    for k, name in enumerate(names):
        folder = OutputFolder(features_dir(dataset, name).parent, name).begin()
        try:
            columns = [f"{name}_d{j}" for j in range(width)]
            np.save(folder.path("values.npy"), values[k])
            folder.write_text("utt_ids.txt", "".join(f"{u.utt_id}\n" for u in utterances))
            folder.write_text("names.txt", "".join(f"{c}\n" for c in columns))
            record = {"dataset": dataset, "data_settings": data_settings, "feature_set": name, "model": model,
                      "repo": REPOS[model], "snapshot": str(snapshot), "commit": snapshot.name,
                      "weights_file": weights.name, "weights_bytes": n_bytes, "weights_sha256": weights_sha,
                      "layer": k, "layers": n_layers,
                      "pooling": ("mean over the layer's frames, whole utterance" if pooling == "mean" else
                                  "mean, then population SD, over the layer's frames, whole utterance, side by side"),
                      "n_utterances": len(utterances), "n_features": width,
                      "names_sha256": sha256_text("\n".join(columns)),
                      "non_finite_values": 0, "sample_rate": rate, "samples_total": int(frames.sum()),
                      "encoder": info, "data_manifest_sha256": manifest_sha,
                      "seconds": round(time.perf_counter() - start, 1)}
            folder.write_json("record.json", record)
            folder.write_json("provenance.json", collect("affect.cli extract-pretrained"))
        except BaseException as err:
            folder.fail(err)
            raise
        written.append(folder.complete())
    log(f"saved {len(written)} feature sets of {model} ({names[0]} to {names[-1]}): {len(utterances)} utterances x "
        f"{width} each, in {time.perf_counter() - start:.0f}s")
    return written


def load_frame_encoder(snapshot: Path, device: str, layers: tuple):
    """(encode, info): encode(int16 samples) -> [float16 [frames, width] for each requested layer], every frame of each
    (added 2026-10-26; the frames heads, affect.models.frames). The model reads the utterance alone, as load_encoder's."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    rate = int(extractor.sampling_rate)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).to(device).eval()

    def encode(samples: np.ndarray) -> list:
        wave = samples.astype(np.float32) / 32768.0
        inputs = extractor(wave, sampling_rate=rate, return_tensors="pt")
        kwargs = {"input_values": inputs["input_values"].to(device)}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(**kwargs, output_hidden_states=True)
        chosen = [out.hidden_states[k][0].float() for k in layers]          # each [frames, width]
        if chosen[0].shape[0] < 1:
            raise ValueError("the encoder gave no frame")
        return [h.cpu().numpy().astype(np.float16) for h in chosen]

    config = model.config
    info = {"model_type": config.model_type, "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size, "layers": config.num_hidden_layers + 1,
            "feature_extractor": {k: getattr(extractor, k, None) for k in
                                  ("sampling_rate", "do_normalize", "return_attention_mask", "padding_value")},
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "device": str(device), "dtype": "float32, stored as float16", "frames_of_layers": list(layers)}
    return encode, info


def extract_frames(dataset: str, data_settings: dict, model: str, layers: tuple, snapshot: Path, device: str, log,
                   encoder=None) -> list:
    """Every utterance of these data settings through the encoder, every frame of the requested layers kept (added
    2026-10-26): one folder artifacts/features/<dataset>/<model>_l<k>_frames/ per layer, written once (frames.npy
    float16 [all frames, width], offsets.npy, utt_ids.txt, record.json, provenance.json; affect.features.frames reads
    them). Values that are not finite in float16 stop it before anything is written."""
    from affect.features.frames import FRAME_RATES

    n_layers, width = PRETRAINED_MODELS[model]
    layers = tuple(int(k) for k in layers)
    if not layers or len(set(layers)) != len(layers) or not all(0 <= k < n_layers for k in layers):
        raise ValueError(f"{model}: layers {list(layers)} must be distinct, from 0 to {n_layers - 1}")
    names = [f"{model}_l{k}_frames" for k in layers]
    existing = [n for n in names if features_dir(dataset, n).exists()]
    if existing:
        raise FileExistsError(f"frames exist already, nothing extracted: {existing}")
    snapshot = Path(snapshot)
    entry = DATASETS.get(dataset)
    settings = validate(entry.settings_class, data_settings, "data")
    utterances = entry.target().utterances(settings)
    rates = {u.sample_rate for u in utterances}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates {rates}")
    rate = rates.pop()
    root = paths.assembled_dir(dataset)
    weights = weights_of(snapshot)
    n_bytes, weights_sha = sha256_file(weights)
    encode, info = encoder if encoder is not None else load_frame_encoder(snapshot, device, layers)
    if info["feature_extractor"].get("sampling_rate") != rate:
        raise ValueError(f"{model} expects {info['feature_extractor'].get('sampling_rate')} Hz, the data is {rate} Hz")
    if info["layers"] != n_layers or info["hidden_size"] != width:
        raise ValueError(f"{model}: the encoder has {info['layers']} layers of {info['hidden_size']}, the table says "
                         f"{n_layers} of {width}")
    start = time.perf_counter()
    chunks = [[] for _ in layers]
    counts = np.empty(len(utterances), dtype=np.int64)
    samples_total = 0
    for i, u in enumerate(utterances):
        samples = read_int16(root / u.audio_path, u.n_frames)
        out = encode(samples)
        if len(out) != len(layers) or any(a.ndim != 2 or a.shape[1] != width or a.shape[0] != out[0].shape[0]
                                          for a in out):
            raise ValueError(f"{u.utt_id}: the encoder gave {[a.shape for a in out]}, expected {len(layers)} x "
                             f"[frames, {width}]")
        for j, a in enumerate(out):
            chunks[j].append(np.asarray(a, dtype=np.float16))
        counts[i] = out[0].shape[0]
        samples_total += samples.shape[0]
        if (i + 1) % 500 == 0 or i + 1 == len(utterances):
            log(f"  {i + 1} of {len(utterances)} utterances, {int(counts[:i + 1].sum()):,} frames, "
                f"{time.perf_counter() - start:.0f}s")
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    stacked = []
    for j, name in enumerate(names):
        values = np.concatenate(chunks[j], axis=0)
        chunks[j] = None
        non_finite = int((~np.isfinite(values)).sum())
        if non_finite:
            raise ValueError(f"{name}: {non_finite} values not finite in float16: nothing written")
        stacked.append(values)
    _, manifest_sha = sha256_file(root / "MANIFEST.csv")
    written = []
    for k, name, values in zip(layers, names, stacked):
        folder = OutputFolder(features_dir(dataset, name).parent, name).begin()
        try:
            np.save(folder.path("frames.npy"), values)
            np.save(folder.path("offsets.npy"), offsets)
            folder.write_text("utt_ids.txt", "".join(f"{u.utt_id}\n" for u in utterances))
            record = {"dataset": dataset, "data_settings": data_settings, "feature_set": name, "model": model,
                      "repo": REPOS[model], "snapshot": str(snapshot), "commit": snapshot.name,
                      "weights_file": weights.name, "weights_bytes": n_bytes, "weights_sha256": weights_sha,
                      "layer": k, "layers": n_layers, "width": width, "dtype": "float16",
                      "frame_rate": FRAME_RATES[model], "n_utterances": len(utterances),
                      "total_frames": int(values.shape[0]), "frames_min": int(counts.min()),
                      "frames_max": int(counts.max()), "non_finite_values": 0, "sample_rate": rate,
                      "samples_total": int(samples_total), "encoder": info, "data_manifest_sha256": manifest_sha,
                      "seconds": round(time.perf_counter() - start, 1)}
            folder.write_json("record.json", record)
            folder.write_json("provenance.json", collect("affect.cli extract-frames"))
        except BaseException as err:
            folder.fail(err)
            raise
        written.append(folder.complete())
    log(f"saved {len(written)} frame sets of {model} ({', '.join(names)}): {len(utterances)} utterances, "
        f"{int(offsets[-1]):,} frames of {width}, in {time.perf_counter() - start:.0f}s")
    return written


def derive_layer_average(dataset: str, model: str, log) -> Path:
    """<model>_lavg: the mean over every layer of the layer's mean vector, from the <model>_l<k> folders (added
    2026-10-20). Every layer's folder must exist, with the same utterances, the same weights and no non-finite value."""
    names = feature_sets(model)
    target = f"{model}_lavg"
    if features_dir(dataset, target).exists():
        raise FileExistsError(f"feature set exists already, nothing derived: {target}")
    records, total, ids = [], None, None
    for name in names:
        folder = features_dir(dataset, name)
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        these = (folder / "utt_ids.txt").read_text(encoding="utf-8")
        values = np.load(folder / "values.npy")
        if ids is None:
            ids = these
        elif these != ids:
            raise ValueError(f"{name}: its utterances are not those of {names[0]}")
        if values.shape != (len(ids.split()), PRETRAINED_MODELS[model][1]) or not np.isfinite(values).all():
            raise ValueError(f"{name}: values of shape {values.shape}, or non-finite values")
        same = ("model", "commit", "weights_sha256", "data_manifest_sha256", "n_utterances")
        if records and any(record.get(k) != records[0].get(k) for k in same):
            raise ValueError(f"{name}: its record differs from {names[0]}'s in {same}")
        records.append(record)
        total = values.astype(np.float64) if total is None else total + values
    average = (total / len(names)).astype(np.float32)
    width = average.shape[1]
    columns = [f"{target}_d{j}" for j in range(width)]
    folder = OutputFolder(features_dir(dataset, target).parent, target).begin()
    try:
        np.save(folder.path("values.npy"), average)
        folder.write_text("utt_ids.txt", ids)
        folder.write_text("names.txt", "".join(f"{c}\n" for c in columns))
        first = records[0]
        record = {k: first.get(k) for k in ("dataset", "data_settings", "model", "repo", "snapshot", "commit",
                                            "weights_file", "weights_bytes", "weights_sha256", "layers",
                                            "n_utterances", "sample_rate", "samples_total", "encoder",
                                            "data_manifest_sha256")}
        record.update({"feature_set": target, "layer": "average",
                       "pooling": "the mean over every layer of the layer's mean over its frames",
                       "derived_from": names, "n_features": width,
                       "names_sha256": sha256_text("\n".join(columns)), "non_finite_values": 0})
        folder.write_json("record.json", record)
        folder.write_json("provenance.json", collect("affect.cli derive-pretrained"))
    except BaseException as err:
        folder.fail(err)
        raise
    done = folder.complete()
    log(f"saved {target}: the mean of {len(names)} layers, {average.shape[0]} utterances x {width}")
    return done


def describe(snapshot: Path) -> dict:
    """The snapshot's config and preprocessor settings, read as JSON (no transformers needed)."""
    snapshot = Path(snapshot)
    return {name: json.loads((snapshot / name).read_text(encoding="utf-8")) for name in CONFIG_FILES
            if (snapshot / name).is_file()}

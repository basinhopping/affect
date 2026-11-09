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

Two encoders whose input is a log-mel spectrogram (added 2026-10-29, workstream C; Dr. Luitel, 2026-10-21: "pretrained
spectrogram classifier or pretrained model on spectrograms"), each reading the spectrogram its own feature extractor
makes from the same samples:

    whisper_large_v3  openai/whisper-large-v3      Whisper large-v3's encoder (Radford et al. 2023): 128 mel bands every
                                                   10 ms, 30 s windows; 32 layers of 1280 and the input embedding (33
                                                   hidden states), 50 frames a second. Trained on transcribed speech, so
                                                   it is a labeled reference beside the self-supervised encoders.
    ast_audioset      MIT/ast-finetuned-audioset-10-10-0.4593  the Audio Spectrogram Transformer (Gong et al. 2021),
                                                   ImageNet's ViT fine-tuned on AudioSet's sound-event labels: Kaldi
                                                   fbank, 128 bands every 10 ms, 1,024 frames (10.24 s) cut into 16 x 16
                                                   patches (strides 10 and 10); 12 layers of 768 and the patch embedding
                                                   (13 hidden states). No transcripts.

Both take a fixed window and pad a shorter input. An utterance is cut into the fewest equal pieces that each fit the
window (one piece for nearly every IEMOCAP utterance), each piece goes through alone, and only the outputs that cover
the piece's own samples are pooled, so the padding's own outputs never enter an utterance's vector (the outputs kept
still attend to it, as the models were trained to): for Whisper the first ceil(samples / 320) encoder frames of the
piece; for AST the patch tokens (not the two summary tokens) whose time span starts within the piece's Kaldi frames,
every frequency row of them. Each layer's vector is the mean (or mean and SD) over those
outputs of all pieces together.

A third encoder whose input is a mel filterbank (added 2026-11-02, week 32; Dr. Luitel, 2026-11-02: every pretrained
spectrogram model, with my heads tuned on each):

    w2v_bert_2        facebook/w2v-bert-2.0        w2v-BERT 2.0 (the speech encoder of Seamless, Barrault et al. 2023):
                                                   a Conformer pretrained on 4.5 million hours of unlabeled speech by
                                                   contrastive and masked prediction; its feature extractor makes 80
                                                   Kaldi filterbank bands every 10 ms, normalizes each band over the
                                                   utterance and stacks frames in pairs (160 values, 50 frames a
                                                   second); 24 layers of 1024 and the feature projection (25 hidden
                                                   states). No transcripts, no window: the whole utterance at once.

Its extractor pads the filterbank to an even number of frames and marks the padded frame in its attention mask; only
the frames the mask marks real are kept (the pair holding the padded frame is left out), for the probes' mean and for
the frames alike.

Whisper's smaller size (added 2026-11-05, RULE_week32_addendum7.txt part S), read exactly as large-v3 is:

    whisper_small     openai/whisper-small         Whisper Small's encoder (Radford et al. 2023): 80 mel bands every
                                                   10 ms, 30 s windows; 12 layers of 768 and the input embedding (13
                                                   hidden states), 50 frames a second. The same training data and
                                                   recipe as the other Whisper sizes, so beside large-v3 it shows what
                                                   the encoder's size alone changes.

Two more of the family (added 2026-11-07, RULE_week32_addendum9.txt part N), read exactly as Whisper Small is:

    bud_e_whisper     laion/BUD-E-Whisper          BUD-E-Whisper (LAION): Whisper Small fine-tuned to caption
                                                   emotional speech (its card: the Laion's Got Talent data and about
                                                   5,000 hours of vlog audio); Whisper Small's shape (13 hidden states
                                                   of 768, 80 mel bands), so beside Whisper Small it shows what that
                                                   fine-tuning changes.
    whisper_medium    openai/whisper-medium        Whisper medium's encoder: 24 layers of 1024 and the input embedding
                                                   (25 hidden states), 80 mel bands; the size between Small and
                                                   large-v3.

Every encoder gives frames for the heads (affect.features.frames; AST's and w2v-BERT 2.0's added 2026-11-02): a waveform
encoder's and w2v-BERT 2.0's frames are its hidden layer's outputs (the real ones), Whisper's the outputs covering each
piece's samples, AST's the patch tokens of each time column averaged over the frequency rows (load_ast_frame_encoder),
10 frames a second. Each frame set's mean over an utterance is the probe's mean vector of the same layer.

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
         "hubert_large": "facebook/hubert-large-ll60k", "wav2vec2_large": "facebook/wav2vec2-large-lv60",
         "whisper_large_v3": "openai/whisper-large-v3", "ast_audioset": "MIT/ast-finetuned-audioset-10-10-0.4593",
         "w2v_bert_2": "facebook/w2v-bert-2.0", "whisper_small": "openai/whisper-small",
         "bud_e_whisper": "laion/BUD-E-Whisper", "whisper_medium": "openai/whisper-medium"}
# the config's model_type of each spectrogram encoder that reads a fixed window (added 2026-10-29); load_encoder reads it
# from the snapshot. w2v-BERT 2.0 ("wav2vec2-bert", added 2026-11-02) reads the whole utterance and goes the general way,
# with its input_features (_model_inputs)
SPECTROGRAM_TYPES = {"whisper": "whisper", "audio-spectrogram-transformer": "ast"}
FILTERBANK_TYPES = ("wav2vec2-bert",)       # reads input_features made from a mel filterbank (added 2026-11-02)
# the model_type a spectrogram encoder's snapshot must have (added 2026-11-02): w2v-BERT 2.0 has the layers and width of
# the Large waveform encoders, so a snapshot of the wrong model would pass every other check
SNAPSHOT_TYPES = {"whisper_large_v3": "whisper", "ast_audioset": "audio-spectrogram-transformer",
                  "w2v_bert_2": "wav2vec2-bert", "whisper_small": "whisper", "bud_e_whisper": "whisper",
                  "whisper_medium": "whisper"}


def check_snapshot(model: str, snapshot: Path) -> None:
    want = SNAPSHOT_TYPES.get(model)
    if want is not None and model_type_of(snapshot) != want:
        raise ValueError(f"{snapshot} holds a {model_type_of(snapshot)!r} model, not {model} ({want!r})")
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


def model_type_of(snapshot: Path) -> str:
    """The model_type in the snapshot's config.json (no transformers needed)."""
    return json.loads((Path(snapshot) / "config.json").read_text(encoding="utf-8")).get("model_type")


def _model_inputs(inputs, device) -> tuple:
    """(the keyword arguments of the model's forward, the number of real frames or None for all) from its feature
    extractor's output for one utterance (added 2026-11-02). A waveform encoder reads input_values; its attention_mask,
    when given, covers samples, and every output frame is kept (None), as before. w2v-BERT 2.0 reads input_features,
    stacked filterbank frames with one output frame each, and its attention_mask marks the real ones, a prefix of ones:
    the count of ones is returned, so that the frame padded to make the filterbank's count even is left out."""
    if "input_features" not in inputs:
        kwargs = {"input_values": inputs["input_values"].to(device)}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"].to(device)
        return kwargs, None
    features = inputs["input_features"]
    mask = inputs.get("attention_mask")
    if mask is None:          # without it the padded pair could not be told apart (_extractor_kwargs asks for it)
        raise ValueError("the feature extractor gave input_features without an attention mask")
    flags = [int(v) for v in mask[0].tolist()]
    real = sum(flags)
    if (mask.shape[0] != 1 or len(flags) != int(features.shape[1]) or real < 1
            or any(v != 1 for v in flags[:real]) or any(v != 0 for v in flags[real:])):
        raise ValueError(f"the feature extractor's attention mask ({len(flags)} frames, {real} marked) is not a prefix "
                         f"of ones over the {int(features.shape[1])} input frames")
    return {"input_features": features.to(device), "attention_mask": mask.to(device)}, real


def _extractor_kwargs(model) -> dict:
    """w2v-BERT 2.0's extractor is always asked for its attention mask, which marks the real frames (whatever its
    preprocessor_config.json says); a waveform encoder's extractor is called as before."""
    return {"return_attention_mask": True} if model.config.model_type in FILTERBANK_TYPES else {}


def _real_layers(hidden_states, real) -> list:
    """Each hidden layer of the one utterance as float32 [frames, width], cut to its real frames when real is given
    (w2v-BERT 2.0's output frames are its input frames, one each, so the mask's count applies to every layer)."""
    layers = [h[0].float() for h in hidden_states]                    # each [frames, width]
    if real is not None:
        if any(h.shape[0] < real for h in layers):
            raise ValueError(f"the encoder gave {[h.shape[0] for h in layers][:2]} frames, fewer than the {real} real")
        layers = [h[:real] for h in layers]
    if layers[0].shape[0] < 1:
        raise ValueError("the encoder gave no frame")
    return layers


def _load_general(snapshot: Path, device: str):
    """(extractor, model in eval mode in float32 on the device, the dtype it was loaded in) for an encoder that reads the
    whole utterance at once: the waveform encoders and w2v-BERT 2.0. transformers 5 loads a checkpoint in its own dtype;
    every weight is cast to float32 (a no-op for a float32 checkpoint, as the waveform encoders' are)."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    loaded_dtype = str(next(model.parameters()).dtype)
    model = model.float().to(device).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("the encoder's weights are not all float32 after the cast")
    return extractor, model, loaded_dtype


def _general_info(extractor, model, device, loaded_dtype) -> dict:
    import torch
    import transformers

    config = model.config
    keys = ("sampling_rate", "do_normalize", "return_attention_mask", "padding_value", "feature_size", "num_mel_bins",
            "stride")
    info = {"model_type": config.model_type, "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size, "layers": config.num_hidden_layers + 1,
            "feature_extractor": {k: getattr(extractor, k, None) for k in keys},
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "device": str(device), "dtype": "float32", "checkpoint_dtype_as_loaded": loaded_dtype}
    if config.model_type in FILTERBANK_TYPES:
        info["input"] = ("mel filterbank from the model's own feature extractor (frames stacked in pairs); the frames "
                         "its attention mask marks real are kept")
    return info


def load_encoder(snapshot: Path, device: str, pooling: str = "mean"):
    """(encode, info): encode(int16 samples) -> float32 [layers, width], the mean of every hidden layer over time; with
    pooling mean_sd, [layers, 2 x width], each layer's mean and then its population SD over time. A spectrogram
    encoder's snapshot that reads a fixed window (its config's model_type in SPECTROGRAM_TYPES) goes to
    load_spectrogram_encoder; w2v-BERT 2.0 (FILTERBANK_TYPES) is pooled over its real frames only."""
    if model_type_of(snapshot) in SPECTROGRAM_TYPES:
        return load_spectrogram_encoder(snapshot, device, pooling)
    import torch

    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; known: {sorted(POOLINGS)}")
    extractor, model, loaded_dtype = _load_general(snapshot, device)
    rate = int(extractor.sampling_rate)
    asks = _extractor_kwargs(model)

    def encode(samples: np.ndarray) -> np.ndarray:
        wave = samples.astype(np.float32) / 32768.0
        inputs = extractor(wave, sampling_rate=rate, return_tensors="pt", **asks)
        kwargs, real = _model_inputs(inputs, device)
        with torch.inference_mode():
            out = model(**kwargs, output_hidden_states=True)
        return pool_layers(_real_layers(out.hidden_states, real), pooling)

    info = _general_info(extractor, model, device, loaded_dtype)
    info["pooling"] = pooling
    return encode, info


def pieces(n: int, window: int) -> list:
    """The fewest pieces of n samples that each fit a window of `window` samples, as equal as whole samples allow
    (lengths differ by at most one): [(start, stop), ...] covering 0 to n once (added 2026-10-29)."""
    if n < 1 or window < 1:
        raise ValueError(f"cannot cut {n} samples into windows of {window}")
    k = -(-n // window)
    bounds = [(i * n) // k for i in range(k + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(k)]


def _whisper_parts(extractor, model, device):
    """Whisper's encoder: (hidden(piece) -> per layer [real frames, width], window samples, layers, width, details)."""
    import torch

    config = model.config
    encoder = model.get_encoder().to(device).eval()
    rate, hop = int(extractor.sampling_rate), int(extractor.hop_length)
    window, mel_frames = int(extractor.n_samples), int(extractor.nb_max_frames)
    positions = int(config.max_source_positions)
    stride = mel_frames // positions                    # mel frames per encoder frame (the second convolution's 2)
    if stride * positions != mel_frames or int(extractor.feature_size) != int(config.num_mel_bins):
        raise ValueError(f"Whisper's extractor ({mel_frames} frames of {extractor.feature_size} bands) does not fit its "
                         f"encoder ({positions} positions, {config.num_mel_bins} bands)")

    def hidden(piece: np.ndarray) -> list:
        inputs = extractor(piece, sampling_rate=rate, return_tensors="pt")
        with torch.inference_mode():
            out = encoder(inputs["input_features"].to(device), output_hidden_states=True)
        real = min(positions, max(1, -(-len(piece) // (hop * stride))))
        return [h[0, :real].float() for h in out.hidden_states]

    details = {"feature_size": int(extractor.feature_size), "hop_length": hop, "window_samples": window,
               "mel_frames_per_encoder_frame": stride, "encoder_positions": positions,
               "outputs_pooled": "the first ceil(samples / (hop_length x mel frames per encoder frame)) encoder "
                                 "frames of each piece"}
    return hidden, window, int(config.encoder_layers) + 1, int(config.d_model), details


def _ast_parts(extractor, model, device):
    """The Audio Spectrogram Transformer: (hidden(piece) -> per layer [real patch tokens, width], window samples,
    layers, width, details). Tokens: two summary tokens, then the patches, frequency row by row (index 2 + f x
    time patches + t)."""
    import torch

    config = model.config
    model = model.to(device).eval()
    rate = int(extractor.sampling_rate)
    max_frames = int(extractor.max_length)
    if max_frames != int(config.max_length) or int(extractor.num_mel_bins) != int(config.num_mel_bins):
        raise ValueError(f"AST's extractor ({max_frames} frames of {extractor.num_mel_bins} bands) does not fit its "
                         f"model ({config.max_length} frames of {config.num_mel_bins} bands)")
    shift, frame = rate // 100, rate * 25 // 1000      # Kaldi's 10 ms shift and 25 ms window: 160 and 400 samples
    window = frame + shift * (max_frames - 1)          # the most samples whose Kaldi frames all fit: 164,080
    f_dim = (int(config.num_mel_bins) - int(config.patch_size)) // int(config.frequency_stride) + 1
    t_dim = (int(config.max_length) - int(config.patch_size)) // int(config.time_stride) + 1

    def hidden(piece: np.ndarray) -> list:
        if len(piece) < frame:
            raise ValueError(f"a piece of {len(piece)} samples is shorter than one Kaldi frame ({frame})")
        inputs = extractor(piece, sampling_rate=rate, return_tensors="pt")
        with torch.inference_mode():
            out = model(inputs["input_values"].to(device), output_hidden_states=True)
        if out.hidden_states[0].shape[1] != 2 + f_dim * t_dim:
            raise ValueError(f"AST gave {out.hidden_states[0].shape[1]} tokens, expected {2 + f_dim * t_dim}")
        n_frames = 1 + (len(piece) - frame) // shift
        real_t = min(t_dim, max(1, -(-n_frames // int(config.time_stride))))
        index = torch.tensor([2 + f * t_dim + t for f in range(f_dim) for t in range(real_t)],
                             device=out.hidden_states[0].device)
        return [h[0].index_select(0, index).float() for h in out.hidden_states]

    details = {"num_mel_bins": int(extractor.num_mel_bins), "max_length": max_frames, "window_samples": window,
               "do_normalize": getattr(extractor, "do_normalize", None), "mean": getattr(extractor, "mean", None),
               "std": getattr(extractor, "std", None), "patches": [f_dim, t_dim],
               "outputs_pooled": "the patch tokens (not the two summary tokens) of every frequency row whose time "
                                 "span starts within the piece's Kaldi frames"}
    return hidden, window, int(config.num_hidden_layers) + 1, int(config.hidden_size), details


def load_spectrogram_encoder(snapshot: Path, device: str, pooling: str = "mean"):
    """(encode, info) for Whisper's encoder or AST (added 2026-10-29): encode(int16 samples) -> float32 [layers,
    width], or [layers, 2 x width] with pooling mean_sd. The utterance is cut into pieces that fit the model's window
    (pieces()); each piece goes through the model's own feature extractor and the model alone; the outputs covering
    the piece's own samples, of all pieces together, are pooled per layer (pool_layers)."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel

    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; known: {sorted(POOLINGS)}")
    extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    # transformers 5 loads a checkpoint in its own dtype, and whisper-large-v3's weights are float16: every weight is
    # cast to float32, so the encoder computes in float32 like the waveform encoders (their checkpoints are float32)
    loaded_dtype = str(next(model.parameters()).dtype)
    model = model.float()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("the encoder's weights are not all float32 after the cast")
    family = SPECTROGRAM_TYPES.get(model.config.model_type)
    if family == "whisper":
        hidden, window, n_layers, width, details = _whisper_parts(extractor, model, device)
    elif family == "ast":
        hidden, window, n_layers, width, details = _ast_parts(extractor, model, device)
    else:
        raise ValueError(f"{snapshot}: model_type {model.config.model_type!r} is not a spectrogram encoder")

    def encode(samples: np.ndarray) -> np.ndarray:
        wave = samples.astype(np.float32) / 32768.0
        per_layer = None
        for start, stop in pieces(len(wave), window):
            layers = hidden(wave[start:stop])
            if len(layers) != n_layers or any(h.ndim != 2 or h.shape[1] != width or h.shape[0] < 1 for h in layers):
                raise ValueError(f"the encoder gave {[tuple(h.shape) for h in layers]}, expected {n_layers} x "
                                 f"[outputs, {width}]")
            per_layer = [[h] for h in layers] if per_layer is None else [p + [h] for p, h in zip(per_layer, layers)]
        return pool_layers([torch.cat(parts, dim=0) for parts in per_layer], pooling)

    info = {"model_type": model.config.model_type, "num_hidden_layers": n_layers - 1, "hidden_size": width,
            "layers": n_layers, "input": "log-mel spectrogram from the model's own feature extractor",
            "feature_extractor": {"sampling_rate": int(extractor.sampling_rate), **details},
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "device": str(device), "dtype": "float32", "checkpoint_dtype_as_loaded": loaded_dtype, "pooling": pooling}
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
    if encoder is None:
        check_snapshot(model, snapshot)
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
    (added 2026-10-26; the frames heads, affect.models.frames). The model reads the utterance alone, as load_encoder's;
    for w2v-BERT 2.0 (added 2026-11-02) only the frames its attention mask marks real, as load_encoder pools them."""
    import torch

    extractor, model, loaded_dtype = _load_general(snapshot, device)
    rate = int(extractor.sampling_rate)
    asks = _extractor_kwargs(model)
    n_layers = int(model.config.num_hidden_layers) + 1
    layers = tuple(int(k) for k in layers)
    if not layers or not all(0 <= k < n_layers for k in layers):
        raise ValueError(f"layers {list(layers)}: the encoder has layers 0 to {n_layers - 1}")

    def encode(samples: np.ndarray) -> list:
        wave = samples.astype(np.float32) / 32768.0
        inputs = extractor(wave, sampling_rate=rate, return_tensors="pt", **asks)
        kwargs, real = _model_inputs(inputs, device)
        with torch.inference_mode():
            out = model(**kwargs, output_hidden_states=True)
        every = _real_layers(out.hidden_states, real)
        return [every[k].cpu().numpy().astype(np.float16) for k in layers]

    info = _general_info(extractor, model, device, loaded_dtype)
    info.update({"dtype": "float32, stored as float16", "frames_of_layers": list(layers)})
    return encode, info


def load_ast_frame_encoder(snapshot: Path, device: str, layers: tuple):
    """(encode, info) for the Audio Spectrogram Transformer (added 2026-11-02, the heads on AST): encode(int16 samples) ->
    [float16 [frames, width] for each requested layer]. The utterance is cut into pieces that fit AST's window
    (pieces()); of each piece, the patch tokens the probe pools (_ast_parts: every frequency row of the time columns
    whose span starts within the piece's Kaldi frames) are averaged over the frequency rows, one frame per time column:
    10 frames a second (the time stride of 10 Kaldi frames of 10 ms), the pieces' frames one after another. Every
    column holds the same number of rows, so the frames' mean over the utterance is the probe's mean vector of the same
    layer (load_spectrogram_encoder)."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    if SPECTROGRAM_TYPES.get(model.config.model_type) != "ast":
        raise ValueError(f"{snapshot}: model_type {model.config.model_type!r} is not AST")
    loaded_dtype = str(next(model.parameters()).dtype)
    model = model.float()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("the encoder's weights are not all float32 after the cast")
    hidden, window, n_layers, width, details = _ast_parts(extractor, model, device)
    f_dim = int(details["patches"][0])
    layers = tuple(int(k) for k in layers)
    if not layers or not all(0 <= k < n_layers for k in layers):
        raise ValueError(f"layers {list(layers)}: the encoder has layers 0 to {n_layers - 1}")
    rate = int(extractor.sampling_rate)
    frame_rate = rate / ((rate // 100) * int(model.config.time_stride))

    def encode(samples: np.ndarray) -> list:
        wave = samples.astype(np.float32) / 32768.0
        kept = [[] for _ in layers]
        for start, stop in pieces(len(wave), window):
            out = hidden(wave[start:stop])
            if len(out) != n_layers or any(h.ndim != 2 or h.shape[1] != width or h.shape[0] < f_dim
                                           or h.shape[0] % f_dim for h in out):
                raise ValueError(f"AST gave {[tuple(h.shape) for h in out]}, expected {n_layers} x [a multiple of "
                                 f"{f_dim} tokens, {width}]")
            for j, k in enumerate(layers):
                h = out[k]                                   # [f_dim x real_t tokens, width], frequency row by row
                kept[j].append(h.reshape(f_dim, h.shape[0] // f_dim, width).mean(dim=0))
        return [torch.cat(parts, dim=0).cpu().numpy().astype(np.float16) for parts in kept]

    info = {"model_type": model.config.model_type, "num_hidden_layers": n_layers - 1, "hidden_size": width,
            "layers": n_layers, "input": "log-mel spectrogram from the model's own feature extractor",
            "feature_extractor": {"sampling_rate": rate, **details},
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "device": str(device), "dtype": "float32, stored as float16", "checkpoint_dtype_as_loaded": loaded_dtype,
            "frames_of_layers": list(layers), "frame_rate": frame_rate,
            "frames_kept": "per time column of each piece, the mean over the frequency rows of its patch tokens (the "
                           "tokens the probe pools), the pieces one after another"}
    return encode, info


def load_whisper_frame_encoder(snapshot: Path, device: str, layers: tuple):
    """(encode, info) for Whisper's encoder (added 2026-11-01, the heads on a spectrogram encoder's frames): encode(int16
    samples) -> [float16 [frames, width] for each requested layer]. The utterance is cut into pieces that fit Whisper's
    30 s window (pieces()), each piece goes through Whisper's own feature extractor and the encoder alone, and the encoder
    frames that cover the piece's own samples (the first ceil(samples / 320) of each piece, as the probes' mean takes
    them, _whisper_parts) are kept, the pieces' frames one after another: 50 frames a second, the padding's outputs left
    out. Their mean over the utterance is the probe's mean vector of the same layer (load_spectrogram_encoder)."""
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
    if SPECTROGRAM_TYPES.get(model.config.model_type) != "whisper":
        raise ValueError(f"{snapshot}: model_type {model.config.model_type!r} is not Whisper")
    loaded_dtype = str(next(model.parameters()).dtype)
    model = model.float()                       # as load_spectrogram_encoder: whisper-large-v3's weights are float16
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("the encoder's weights are not all float32 after the cast")
    hidden, window, n_layers, width, details = _whisper_parts(extractor, model, device)
    layers = tuple(int(k) for k in layers)
    if not layers or not all(0 <= k < n_layers for k in layers):
        raise ValueError(f"layers {list(layers)}: the encoder has layers 0 to {n_layers - 1}")

    def encode(samples: np.ndarray) -> list:
        wave = samples.astype(np.float32) / 32768.0
        kept = [[] for _ in layers]
        for start, stop in pieces(len(wave), window):
            out = hidden(wave[start:stop])
            if len(out) != n_layers or any(h.ndim != 2 or h.shape[1] != width or h.shape[0] < 1 for h in out):
                raise ValueError(f"the encoder gave {[tuple(h.shape) for h in out]}, expected {n_layers} x "
                                 f"[frames, {width}]")
            for j, k in enumerate(layers):
                kept[j].append(out[k])
        return [torch.cat(parts, dim=0).cpu().numpy().astype(np.float16) for parts in kept]

    info = {"model_type": model.config.model_type, "num_hidden_layers": n_layers - 1, "hidden_size": width,
            "layers": n_layers, "input": "log-mel spectrogram from the model's own feature extractor",
            "feature_extractor": {"sampling_rate": int(extractor.sampling_rate), **details},
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "device": str(device), "dtype": "float32, stored as float16", "checkpoint_dtype_as_loaded": loaded_dtype,
            "frames_of_layers": list(layers),
            "frames_kept": "the first ceil(samples / 320) encoder frames of each piece, the pieces one after another"}
    return encode, info


def extract_frames(dataset: str, data_settings: dict, model: str, layers: tuple, snapshot: Path, device: str, log,
                   encoder=None) -> list:
    """Every utterance of these data settings through the encoder, every frame of the requested layers kept (added
    2026-10-26): one folder artifacts/features/<dataset>/<model>_l<k>_frames/ per layer, written once (frames.npy
    float16 [all frames, width], offsets.npy, utt_ids.txt, record.json, provenance.json; affect.features.frames reads
    them). Values that are not finite in float16 stop it before anything is written."""
    from affect.features.frames import FRAME_RATES

    if model not in FRAME_RATES:
        raise ValueError(f"{model} gives no frames here; its layers are probed with extract")
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
    if encoder is None:
        check_snapshot(model, snapshot)
        family = SPECTROGRAM_TYPES.get(model_type_of(snapshot))      # the snapshot's own config decides (2026-11-02)
        loader = {"whisper": load_whisper_frame_encoder, "ast": load_ast_frame_encoder}.get(family, load_frame_encoder)
        encoder = loader(snapshot, device, layers)
    encode, info = encoder
    if info["feature_extractor"].get("sampling_rate") != rate:
        raise ValueError(f"{model} expects {info['feature_extractor'].get('sampling_rate')} Hz, the data is {rate} Hz")
    if info["layers"] != n_layers or info["hidden_size"] != width:
        raise ValueError(f"{model}: the encoder has {info['layers']} layers of {info['hidden_size']}, the table says "
                         f"{n_layers} of {width}")
    if "frame_rate" in info and abs(float(info["frame_rate"]) - FRAME_RATES[model]) > 1e-9:
        raise ValueError(f"{model}: the encoder gives {info['frame_rate']} frames a second, the table says "
                         f"{FRAME_RATES[model]}")
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
